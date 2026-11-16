#!/usr/bin/env bash
# Start a headless llama.cpp build on the Pi from the PC.
#
# Copies the Pi scripts to ~/bonsai/scripts on the Pi and starts
# build-llama-cpp.sh as a detached job, so the build carries on if the SSH
# connection drops. Progress is logged to ~/bonsai/jobs/build-llama-cpp/log.txt.
#
# Usage: BONSAI_PI=user@host scripts/pi/remote-build.sh [ref]
#   ref defaults to the version pinned in configs/llama-cpp.ref.

set -euo pipefail

pi=${BONSAI_PI:?set BONSAI_PI to the SSH destination of the Pi, e.g. user@host}
here=$(cd "$(dirname "$0")" && pwd)
pinned_ref=$(<"$here/../../configs/llama-cpp.ref")
ref=${1:-$pinned_ref}

ssh "$pi" 'mkdir -p ~/bonsai/scripts'
scp -q "$here/run-detached.sh" "$here/build-llama-cpp.sh" "$pi:bonsai/scripts/"
# shellcheck disable=SC2029  # $ref is meant to expand on this side
ssh "$pi" "chmod +x ~/bonsai/scripts/*.sh && \
    ~/bonsai/scripts/run-detached.sh build-llama-cpp ~/bonsai/jobs/build-llama-cpp -- \
    ~/bonsai/scripts/build-llama-cpp.sh $(printf %q "$ref")"

echo "follow progress with: ssh $pi tail -f bonsai/jobs/build-llama-cpp/log.txt"
