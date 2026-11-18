#!/usr/bin/env bash
cd /home/callum/bonsai/jobs/bench-qwen-single-20261118T120724Z
date -u +%Y-%m-%dT%H:%M:%SZ > started_at
env THREADS=3 REPACK=0 REFERENCE=\*single4-q4_k_m\* /home/callum/bonsai/scripts/pi/bench-trees.sh inputs results > log.txt 2>&1
status=$?
date -u +%Y-%m-%dT%H:%M:%SZ > finished_at
echo "$status" > exit_code
