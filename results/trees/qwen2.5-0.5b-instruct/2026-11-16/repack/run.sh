#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
bin=~/bonsai/llama.cpp/build/bin
trees=~/bonsai/trees/qwen2.5-0.5b-instruct
for q in q4_0 q8_0; do
  for rp in 0 1; do
    echo "== $q repack=$rp"
    python3 ~/bonsai/scripts/pi/peak_rss.py rss.tmp -- "$bin/llama-bench" -m "$trees/qwen2.5-0.5b-instruct-$q.gguf" \
      --repack "$rp" -t 4 -p 128 -n 128 -r 3 -o jsonl >> results/repack.jsonl
    printf "%s\t%s\t%s\n" "$q" "$rp" "$(< rss.tmp)" >> results/repack-rss.tsv
    sleep 10
  done
done
echo "== done"
