#!/usr/bin/env bash
cd /home/callum/bonsai/jobs/bench-qwen-pot-20261118T112522Z
date -u +%Y-%m-%dT%H:%M:%SZ > started_at
env THREADS=3 REPACK=0 REFERENCE=\*q4_0\* /home/callum/bonsai/scripts/pi/bench-trees.sh inputs results > log.txt 2>&1
status=$?
date -u +%Y-%m-%dT%H:%M:%SZ > finished_at
echo "$status" > exit_code
