#!/usr/bin/env python3
import random
import ollama
from collections import defaultdict

MODEL = "codellama:instruct"
SHOT_LIST = [2, 4, 8]
DATASET_SIZE = 120
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
GLOBAL_SEED = 0

def generate_roofline_dataset(n):
    ds = []
    for _ in range(n):
        pg = random.uniform(1e3, 1e4)
        pbw = random.uniform(1e2, 1e3)
        bal = pg / pbw

        # compute-bound examples
        ai_cb = bal * random.uniform(1.1, 2.0)
        perf_cb = ai_cb * pbw
        ds.append({"peak_gflops": pg, "peak_bw": pbw, "ai": ai_cb, "perf": perf_cb, "label": "Compute"})

        # bandwidth-bound examples
        ai_bb = bal * random.uniform(0.1, 0.9)
        perf_bb = ai_bb * pbw
        ds.append({"peak_gflops": pg, "peak_bw": pbw, "ai": ai_bb, "perf": perf_bb, "label": "Bandwidth"})
    return ds

def stratified_split(dataset, ratios, seed=0):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in dataset:
        groups[s["label"]].append(s)
    for lbl in groups:
        rng.shuffle(groups[lbl])

    train, val, test = [], [], []
    for lbl, items in groups.items():
        n = len(items)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        n_test = n - n_train - n_val
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test

def _sample_balanced(pool, k, rng):
    comp = [x for x in pool if x["label"] == "Compute"]
    band = [x for x in pool if x["label"] == "Bandwidth"]
    take_c = min(len(comp), k // 2)
    take_b = min(len(band), k // 2)
    shots = []
    shots += rng.sample(comp, take_c) if len(comp) >= take_c else comp
    shots += rng.sample(band, take_b) if len(band) >= take_b else band
    # filling any remainder from the pooled list
    leftover = [x for x in pool if x not in shots]
    need = k - len(shots)
    if need > 0 and leftover:
        shots += rng.sample(leftover, min(need, len(leftover)))
    return shots

def build_prompt(sample, shots, cot=False, support_pool=None, support_seed=42):
    rng = random.Random(support_seed)

    examples = []
    if support_pool:
        examples = _sample_balanced(support_pool, min(shots, len(support_pool)), rng)
    else:
        for _ in range(shots):
            pg_i = random.uniform(1e3, 1e4)
            pbw_i = random.uniform(1e2, 1e3)
            bal_i = pg_i / pbw_i
            lbl_i = rng.choice(["Compute", "Bandwidth"])
            ai_i = bal_i * (rng.uniform(1.1, 2.0) if lbl_i == "Compute" else rng.uniform(0.1, 0.9))
            perf_i = ai_i * pbw_i
            examples.append({"peak_gflops": pg_i, "peak_bw": pbw_i, "ai": ai_i, "perf": perf_i, "label": lbl_i})

    lines = []
    for idx, ex in enumerate(examples, start=1):
        pg_i, pbw_i, ai_i, perf_i, lbl_i = ex["peak_gflops"], ex["peak_bw"], ex["ai"], ex["perf"], ex["label"]
        bal_i = pg_i / pbw_i
        lines.append(f"{'CoT' if cot else 'No-CoT'} example {idx} (shown below):")
        lines.append(
            f"Question: Given a GPU having a global memory with a max bandwidth of {pbw_i:.2f} GB/s "
            f"and a peak performance of {pg_i:.2f} GFLOP/s, if a program executed with an Arithmetic Intensity "
            f"of {ai_i:.2f} FLOP/Byte and a performance of {perf_i:.2f} GFLOP/s, does the roofline model consider "
            f"the program as compute-bound or bandwidth-bound?"
        )
        if cot:
            lines.append(
                f"Thought: The balance point is {pg_i:.2f}/{pbw_i:.2f} = {bal_i:.2f} FLOP/Byte. "
                f"The program’s Arithmetic Intensity is {ai_i:.2f} FLOP/Byte. "
                f"Because {ai_i:.2f} {'>' if lbl_i == 'Compute' else '<'} {bal_i:.2f}, it is "
                f"{'after' if lbl_i == 'Compute' else 'before'} the balance point, i.e., {lbl_i.lower()}-bound."
            )
        lines.append(f"Answer: {lbl_i}")
        lines.append("")

    pg = sample["peak_gflops"]; pbw = sample["peak_bw"]; ai = sample["ai"]; perf = sample["perf"]
    lines.append(
        f"Question: Given a GPU having a global memory with a max bandwidth of {pbw:.2f} GB/s "
        f"and a peak performance of {pg:.2f} GFLOP/s, if a program executed with an Arithmetic Intensity "
        f"of {ai:.2f} FLOP/Byte and a performance of {perf:.2f} GFLOP/s, does the roofline model consider "
        f"the program as compute-bound or bandwidth-bound?"
    )
    lines.append("Answer:")
    return "\n".join(lines)

def evaluate(dataset, shots, cot, support_pool):
    correct = 0
    total = len(dataset)
    for idx, sample in enumerate(dataset, start=1):
        prompt = build_prompt(sample, shots, cot=cot, support_pool=support_pool,
                              support_seed=hash((shots, cot, idx)) & 0xFFFFFFFF)
        resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        out = resp["message"]["content"].strip().lower()
        ok = sample["label"].lower() in out
        correct += int(ok)
        print(f"{idx}/{total} | Gold: {sample['label']} | Pred: {out[:40]!r} | {'Correct' if ok else 'Incorrect'}")
    return correct / total if total else 0.0

def evaluate_split(name, dataset, train_pool):
    print(f"\n=== {name} set (n={len(dataset)}) ===")
    for shots in SHOT_LIST:
        print(f"\n{shots}-shot evaluation (No-CoT):")
        acc_nc = evaluate(dataset, shots, cot=False, support_pool=train_pool)
        print(f" {shots}-shot No-CoT Acc.: {acc_nc*100:.2f}%")
        print(f"\n{shots}-shot evaluation (CoT):")
        acc_c = evaluate(dataset, shots, cot=True, support_pool=train_pool)
        print(f" {shots}-shot CoT Acc.: {acc_c*100:.2f}%")

def main():
    random.seed(GLOBAL_SEED)
    ds = generate_roofline_dataset(DATASET_SIZE)

    train, val, test = stratified_split(ds, SPLIT_RATIOS, seed=GLOBAL_SEED)
    print(f"Sizes -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    evaluate_split("Validation", val, train_pool=train)
    evaluate_split("Test", test, train_pool=train)

if __name__ == "__main__":
    main()
