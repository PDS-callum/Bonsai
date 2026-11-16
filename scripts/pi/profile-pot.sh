#!/usr/bin/env bash
# Profile the pot: hardware facts, memory bandwidth and thermals.
#
# Writes into <out-dir>:
#   facts-before.json   hardware, memory, storage and thermal snapshot
#   membw.jsonl         bandwidth per kernel, per thread count, every sample
#   telemetry.csv       temperature, clock, throttling, load and free memory,
#                       sampled every second while the benchmarks run
#   facts-after.json    the same snapshot once the benchmarks finish
#
# Meant to run detached through run-detached.sh. Uses about 3 x SIZE_MIB of
# memory (768 MiB by default), so check the pot's free memory first.
#
# Usage: profile-pot.sh <out-dir>
# Environment: SIZE_MIB (default 256), REPEATS (default 10),
#              NOISE_REPEATS (default 30), BONSAI_HOME (default ~/bonsai)

set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $(basename "$0") <out-dir>" >&2; exit 2; }
out_dir=$1
size_mib=${SIZE_MIB:-256}
repeats=${REPEATS:-10}
noise_repeats=${NOISE_REPEATS:-30}
home=${BONSAI_HOME:-$HOME/bonsai}
script_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$out_dir" "$home/tools/bin"
membw="$home/tools/bin/membw"

echo "== building membw"
gcc -O3 -march=native -fopenmp -Wall -Wextra -o "$membw" "$script_dir/membw.c"

echo "== facts before"
python3 "$script_dir/pot_facts.py" "$home" > "$out_dir/facts-before.json"

"$script_dir/telemetry.sh" > "$out_dir/telemetry.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

# A short idle window first gives a baseline for temperature and clock.
sleep 10

: > "$out_dir/membw.jsonl"
for threads in 1 2 3 4; do
    echo "== membw: $threads thread(s), ${size_mib} MiB, $repeats repeats"
    OMP_NUM_THREADS=$threads OMP_PROC_BIND=close \
        "$membw" "$size_mib" "$repeats" | tee -a "$out_dir/membw.jsonl"
done

# A longer run at full width shows how much the pot's neighbours move results.
echo "== membw: noise run, 4 threads, $noise_repeats repeats"
OMP_NUM_THREADS=4 OMP_PROC_BIND=close \
    "$membw" "$size_mib" "$noise_repeats" | tee -a "$out_dir/membw.jsonl"

# And a cool-down window to see the temperature settle.
sleep 10

echo "== facts after"
python3 "$script_dir/pot_facts.py" "$home" > "$out_dir/facts-after.json"
echo "== done"
