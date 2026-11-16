#!/usr/bin/env bash
# Build llama.cpp for CPU-only inference on the Raspberry Pi.
#
# Clones llama.cpp into $BONSAI_HOME/llama.cpp, checks out the requested ref,
# and builds the inference tools (llama-cli, llama-bench, llama-server) with
# native ARM optimisations. Writes build-info.json next to the binaries.
#
# Needs git, a C/C++ compiler and python3. If cmake isn't installed, it is
# installed from PyPI into $BONSAI_HOME/tools, so no root access is needed.
#
# Usage: build-llama-cpp.sh <ref> [expected-commit]
#   If expected-commit is given, the build stops unless <ref> resolves to it.
# Environment:
#   BONSAI_HOME  working directory on the Pi (default: ~/bonsai)
#   JOBS         parallel compile jobs (default: 2, to leave RAM for other
#                services on a shared Pi)

set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || {
    echo "usage: $(basename "$0") <ref> [expected-commit]" >&2
    exit 2
}
ref=$1
expected_commit=${2:-}
home=${BONSAI_HOME:-$HOME/bonsai}
jobs=${JOBS:-2}
src="$home/llama.cpp"
repo_url=https://github.com/ggml-org/llama.cpp.git

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if ! command -v cmake >/dev/null; then
    venv="$home/tools/venv"
    if [[ ! -x $venv/bin/cmake ]]; then
        log "installing cmake into $venv"
        python3 -m venv "$venv"
        "$venv/bin/pip" install --quiet --upgrade pip
        "$venv/bin/pip" install --quiet "cmake>=3.28,<5"
    fi
    export PATH="$venv/bin:$PATH"
fi
log "using $(cmake --version | head -n1)"

if [[ ! -d $src/.git ]]; then
    log "cloning $repo_url"
    git clone --filter=blob:none --quiet "$repo_url" "$src"
fi
git -C "$src" fetch --quiet --tags origin
git -C "$src" -c advice.detachedHead=false checkout --quiet "$ref"
commit=$(git -C "$src" rev-parse HEAD)
if [[ -n $expected_commit && $commit != "$expected_commit" ]]; then
    echo "$ref resolved to $commit, expected $expected_commit" >&2
    exit 1
fi
log "checked out $ref ($commit)"

cmake_flags=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_NATIVE=ON
    -DLLAMA_OPENSSL=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_USE_PREBUILT_UI=OFF # only the server's HTTP API is used, not its web UI
)

log "configuring"
cmake -S "$src" -B "$src/build" "${cmake_flags[@]}"

log "building with $jobs jobs"
start=$(date +%s)
cmake --build "$src/build" --config Release -j "$jobs" \
    --target llama-cli llama-bench llama-server
elapsed=$(( $(date +%s) - start ))
log "build finished in ${elapsed}s"

bin="$src/build/bin"
"$bin/llama-cli" --version

python3 - "$bin/build-info.json" "$ref" "$commit" "$elapsed" "${cmake_flags[@]}" <<'EOF'
import json, platform, subprocess, sys
from datetime import datetime, timezone

out, ref, commit, elapsed, *flags = sys.argv[1:]

def first_line(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]

info = {
    "ref": ref,
    "commit": commit,
    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "build_seconds": int(elapsed),
    "machine": platform.machine(),
    "kernel": platform.release(),
    "compiler": first_line("cc", "--version"),
    "cmake": first_line("cmake", "--version"),
    "cmake_flags": flags,
}
with open(out, "w") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
print(f"wrote {out}")
EOF
