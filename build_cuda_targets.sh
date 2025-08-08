#!/bin/bash

SRC_DIR=~/hecBench/HeCBench/src_2
BUILD_DIR=~/hecBench/HeCBench/build
RESULT_DIR=~/hecBench/HeCBench

SUCCESS_FILE="$RESULT_DIR/successful_cuda_targets_2.txt"
FAILED_FILE="$RESULT_DIR/failed_cuda_targets_2.txt"
TIMEOUT=40  # seconds per target

# Empty the files at start
> "$SUCCESS_FILE"
> "$FAILED_FILE"

mkdir -p "$BUILD_DIR"
cd "$SRC_DIR" || { echo "Source directory not found!"; exit 1; }

SUCCESSFUL_BUILDS=()
FAILED_BUILDS=()

for dir in *-cuda/; do
  [ -d "$dir" ] || continue
  echo "------------------------------"
  echo "Building CUDA target: $dir (timeout: ${TIMEOUT}s)"
  cd "$dir"
  if [ $? -ne 0 ]; then
    FAILED_BUILDS+=("$dir: could not enter directory")
    echo "$dir : FAILED (could not enter directory)" >> "$FAILED_FILE"
    continue
  fi

  if timeout "${TIMEOUT}"s make run; then
    exe_file=$(find . -maxdepth 1 -type f -executable ! -name "*.*" | head -n 1)
    if [ -n "$exe_file" ]; then
      exe_basename="$(basename "$dir")"
      mv -v "$exe_file" "$BUILD_DIR/$exe_basename"
      echo "✅ SUCCESS: $dir"
      SUCCESSFUL_BUILDS+=("$dir")
      echo "$dir" >> "$SUCCESS_FILE"
    else
      echo "⚠️  No executable found to move for $dir"
      FAILED_BUILDS+=("$dir: no executable found")
      echo "$dir : FAILED (no executable found)" >> "$FAILED_FILE"
    fi
  else
    # Figure out if it was a timeout or just failed build
    if [ $? -eq 124 ]; then
      echo "❌ FAILED (timed out): $dir"
      FAILED_BUILDS+=("$dir: timed out")
      echo "$dir : FAILED (timed out)" >> "$FAILED_FILE"
    else
      echo "❌ FAILED: $dir"
      FAILED_BUILDS+=("$dir: build failed")
      echo "$dir : FAILED (build failed)" >> "$FAILED_FILE"
    fi
  fi

  cd "$SRC_DIR"
done

echo "------------------------------"
echo "SUCCESSFUL BUILDS: ${#SUCCESSFUL_BUILDS[@]}"
printf "  %s\n" "${SUCCESSFUL_BUILDS[@]}"
echo "FAILED/TIMED‑OUT: ${#FAILED_BUILDS[@]}"
printf "  %s\n" "${FAILED_BUILDS[@]}"
echo "Executables moved to: $BUILD_DIR"
echo
echo "See $SUCCESS_FILE and $FAILED_FILE for results."
