#!/usr/bin/env bash
# Measure the speed, memory and heat of every tree (GGUF) in a directory.
#
# For each tree, llama-bench measures prompt processing (pp) and token
# generation (tg) speed over several repetitions, and its peak resident memory
# is recorded. Then two extra runs use one reference tree:
#
#   threads    tg speed at 1-4 threads, to see whether cores or memory limit it
#   sustained  a long generation run, to see whether heat lowers the clock
#
# Telemetry (temperature, clock, throttling, load, free memory) is sampled every
# second throughout.
#
# Writes into <out-dir>:
#   facts-before.json, facts-after.json   pot snapshots
#   bench.jsonl                           one llama-bench record per test
#   peak-rss.tsv                          tree file name, peak RSS in bytes
#   threads.jsonl, sustained.jsonl        the extra runs
#   telemetry.csv                         sampled every second
#   build-info.json                       the llama.cpp build that ran
#
# Usage: bench-trees.sh <tree-dir> <out-dir>
# Environment: THREADS (default 4), REPACK (llama.cpp weight repacking, 1 or 0;
#              default 1), REPEATS (5), PROMPT_TOKENS (128), GEN_TOKENS (128),
#              SUSTAINED_TOKENS (1024), REFERENCE (file-name pattern for the
#              reference tree, default *q4_k_m*), BONSAI_HOME (~/bonsai)

set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $(basename "$0") <tree-dir> <out-dir>" >&2; exit 2; }
tree_dir=$(cd "$1" && pwd)
out_dir=$2
threads=${THREADS:-4}
repack=${REPACK:-1}
repeats=${REPEATS:-5}
prompt_tokens=${PROMPT_TOKENS:-128}
gen_tokens=${GEN_TOKENS:-128}
sustained_tokens=${SUSTAINED_TOKENS:-1024}
reference=${REFERENCE:-*q4_k_m*}
home=${BONSAI_HOME:-$HOME/bonsai}
script_dir=$(cd "$(dirname "$0")" && pwd)
bin="$home/llama.cpp/build/bin"

shopt -s nullglob
trees=("$tree_dir"/*.gguf)
[[ ${#trees[@]} -gt 0 ]] || { echo "no .gguf files in $tree_dir" >&2; exit 1; }
# The pattern is meant to glob, so it stays unquoted.
# shellcheck disable=SC2206
reference_trees=("$tree_dir"/$reference.gguf)
[[ ${#reference_trees[@]} -eq 1 ]] \
    || { echo "need exactly one reference tree matching $reference" >&2; exit 1; }
reference_tree=${reference_trees[0]}

mkdir -p "$out_dir"
cp "$bin/build-info.json" "$out_dir/"
python3 "$script_dir/pot_facts.py" "$home" > "$out_dir/facts-before.json"

"$script_dir/telemetry.sh" > "$out_dir/telemetry.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

# A short idle window gives a baseline for temperature and clock.
sleep 10

: > "$out_dir/bench.jsonl"
: > "$out_dir/peak-rss.tsv"
for tree in "${trees[@]}"; do
    name=$(basename "$tree")
    echo "== $name: pp$prompt_tokens and tg$gen_tokens, $threads threads," \
        "repack $repack, $repeats repetitions"
    python3 "$script_dir/peak_rss.py" "$out_dir/rss.tmp" -- \
        "$bin/llama-bench" -m "$tree" -t "$threads" --repack "$repack" \
        -p "$prompt_tokens" -n "$gen_tokens" -r "$repeats" -o jsonl \
        >> "$out_dir/bench.jsonl"
    printf '%s\t%s\n' "$name" "$(< "$out_dir/rss.tmp")" >> "$out_dir/peak-rss.tsv"
    rm -f "$out_dir/rss.tmp"
    # Let the pot cool a little so each tree starts from a similar state.
    sleep 15
done

echo "== threads: $(basename "$reference_tree"), tg$gen_tokens at 1-4 threads"
"$bin/llama-bench" -m "$reference_tree" -t 1,2,3,4 --repack "$repack" \
    -p 0 -n "$gen_tokens" -r "$repeats" -o jsonl > "$out_dir/threads.jsonl"
sleep 15

echo "== sustained: $(basename "$reference_tree"), tg$sustained_tokens x 3"
"$bin/llama-bench" -m "$reference_tree" -t "$threads" --repack "$repack" \
    -p 0 -n "$sustained_tokens" -r 3 -o jsonl > "$out_dir/sustained.jsonl"

# A cool-down window shows how quickly the temperature settles.
sleep 30
python3 "$script_dir/pot_facts.py" "$home" > "$out_dir/facts-after.json"
echo "== done"
