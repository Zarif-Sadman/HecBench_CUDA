#!/usr/bin/env python3
import random
import ollama

MODEL = "codellama:instruct"
SHOT_LIST = [2, 4, 8]
DATASET_SIZE = 120

def generate_roofline_dataset(n):
    ds = []
    for _ in range(n):
        pg = random.uniform(1e3, 1e4)
        pbw = random.uniform(1e2, 1e3)
        bal = pg / pbw
        ai_cb = bal * random.uniform(1.1, 2.0)
        perf_cb = ai_cb * pbw
        ds.append({
            "peak_gflops": pg,
            "peak_bw": pbw,
            "ai": ai_cb,
            "perf": perf_cb,
            "label": "Compute"
        })
        ai_bb = bal * random.uniform(0.1, 0.9)
        perf_bb = ai_bb * pbw
        ds.append({
            "peak_gflops": pg,
            "peak_bw": pbw,
            "ai": ai_bb,
            "perf": perf_bb,
            "label": "Bandwidth"
        })
    return ds


def build_prompt(sample, shots, cot=False):
    random.seed(42)
    pool = []
    for _ in range(shots):
        pg_i = random.uniform(1e3, 1e4)
        pbw_i = random.uniform(1e2, 1e3)
        bal_i = pg_i / pbw_i
        label_i = random.choice(["Compute", "Bandwidth"])
        ai_i = bal_i * (random.uniform(1.1, 2.0) if label_i == "Compute" else random.uniform(0.1, 0.9))
        perf_i = ai_i * pbw_i
        pool.append((pg_i, pbw_i, ai_i, perf_i, label_i))

    lines = []
    if cot:
        pg_i, pbw_i, ai_i, perf_i, lbl_i = pool[0]
        bal_i = pg_i / pbw_i
        lines.append("Chain-of-Thought (CoT) Prompt Example")
        lines.append("")
        lines.append("CoT example 1 (shown below):")
        lines.append(
            f"Question: Given a GPU having a global memory with a max bandwidth of {pbw_i:.2f} GB/s and a peak performance of {pg_i:.2f} GFLOP/s, "
            f"if a program executed with an Arithmetic Intensity of {ai_i:.2f} FLOP/Byte and a performance of {perf_i:.2f} GFLOP/s, "
            f"does the roofline model consider the program as compute-bound or bandwidth-bound?"
        )
        lines.append(
            f"Thought: The max bandwidth is {pbw_i:.2f} GB/s, and peak performance is {pg_i:.2f} GFLOP/s. "
            f"The balance point is {pg_i:.2f} / {pbw_i:.2f} = {bal_i:.2f} FLOP/Byte. "
            f"The program’s Arithmetic Intensity is {ai_i:.2f} FLOP/Byte. "
            f"Because {ai_i:.2f} {'>' if lbl_i == 'Compute' else '<'} {bal_i:.2f}, it is {'after' if lbl_i == 'Compute' else 'before'} "
            f"the balance point, putting the program in the {lbl_i.lower()}-bound region. "
            f"The roofline model would consider the program as {lbl_i.lower()}-bound."
        )
        lines.append(f"Answer: {lbl_i}")
        lines.append("")
        lines.append(f"CoT examples 2-{shots} [redacted]")
        lines.append("")
    else:
        for pg_i, pbw_i, ai_i, perf_i, lbl_i in pool:
            lines.append(f"Hardware: {pg_i:.2f} GFLOPS, {pbw_i:.2f} GB/s")
            lines.append(f"Arithmetic intensity: {ai_i:.2f}")
            lines.append(f"Answer: {lbl_i}")
            lines.append("")

    pg = sample["peak_gflops"]
    pbw = sample["peak_bw"]
    ai = sample["ai"]
    perf = sample["perf"]
    lines.append(
        f"Question: Given a GPU having a global memory with a max bandwidth of {pbw:.2f} GB/s and a peak performance of {pg:.2f} GFLOP/s, "
        f"if a program executed with an Arithmetic Intensity of {ai:.2f} FLOP/Byte and a performance of {perf:.2f} GFLOP/s, "
        f"does the roofline model consider the program as compute-bound or bandwidth-bound?"
    )
    lines.append("Answer:")

    return "\n".join(lines)


def evaluate(dataset, shots, cot=False):
    correct = 0
    total = len(dataset)

    for idx, sample in enumerate(dataset, start=1):
        prompt = build_prompt(sample, shots, cot)
        resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        out = resp["message"]["content"].strip().lower()
        if sample["label"].lower() in out:
            correct += 1
        status = 'Correct' if sample["label"].lower() in out else 'Incorrect'
        print(f"{idx}/{total} | Gold: {sample['label']} | Pred: {out[:30]!r} | {status}")

    return correct / total


def main():
    random.seed(0)
    ds = generate_roofline_dataset(DATASET_SIZE)

    for shots in SHOT_LIST:
        print(f"\n{shots}-shot evaluation (No-CoT):")
        acc_nc = evaluate(ds, shots, cot=False)
        print(f" {shots}-shot No-CoT Acc.: {acc_nc*100:.2f}%")

        print(f"\n{shots}-shot evaluation (CoT):")
        acc_c = evaluate(ds, shots, cot=True)
        print(f" {shots}-shot CoT Acc.: {acc_c*100:.2f}%")

if __name__ == "__main__":
    main()
