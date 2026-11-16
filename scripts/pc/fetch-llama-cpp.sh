#!/usr/bin/env bash
# Set up llama.cpp on the PC (the potting bench) at the pinned version.
#
# - Clones the source at LLAMA_CPP_REF into $BONSAI_HOME/llama.cpp, for the
#   Python conversion script (convert_hf_to_gguf.py) and its gguf library.
# - On Windows x64, downloads the official prebuilt CPU binaries for the same
#   commit (llama-quantize and friends) into $BONSAI_HOME/llama.cpp-bin/<build>
#   and verifies their SHA-256 checksum.
#
# Usage: scripts/pc/fetch-llama-cpp.sh
# Environment:
#   BONSAI_HOME  working directory (default: ~/bonsai)

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=SCRIPTDIR/../../configs/llama-cpp.env
source "$here/../../configs/llama-cpp.env"

home=${BONSAI_HOME:-$HOME/bonsai}
src="$home/llama.cpp"
repo_url=https://github.com/ggml-org/llama.cpp

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [[ ! -d $src/.git ]]; then
    log "cloning $repo_url"
    git clone --filter=blob:none --quiet "$repo_url.git" "$src"
fi
git -C "$src" fetch --quiet --tags origin
git -C "$src" -c advice.detachedHead=false checkout --quiet "$LLAMA_CPP_REF"
commit=$(git -C "$src" rev-parse HEAD)
if [[ $commit != "$LLAMA_CPP_COMMIT" ]]; then
    echo "$LLAMA_CPP_REF resolved to $commit, expected $LLAMA_CPP_COMMIT" >&2
    exit 1
fi
log "source at $LLAMA_CPP_REF ($commit) in $src"

case "$(uname -s)-$(uname -m)" in
    MINGW64*-x86_64 | MSYS*-x86_64 | CYGWIN*-x86_64) ;;
    *)
        log "no prebuilt binaries configured for this platform;"
        log "build from source instead, e.g. with scripts/pi/build-llama-cpp.sh"
        exit 0
        ;;
esac

asset="llama-$LLAMA_CPP_BUILD-bin-win-cpu-x64.zip"
bin="$home/llama.cpp-bin/$LLAMA_CPP_BUILD"
if [[ -x $bin/llama-quantize.exe ]]; then
    log "binaries already present in $bin"
else
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    log "downloading $asset"
    curl -fsSL -o "$tmp/$asset" \
        "$repo_url/releases/download/$LLAMA_CPP_BUILD/$asset"
    echo "$LLAMA_CPP_WIN_CPU_X64_SHA256  $tmp/$asset" | sha256sum --check --quiet
    log "checksum verified"
    mkdir -p "$bin"
    unzip -q -o "$tmp/$asset" -d "$bin"
fi

"$bin/llama-cli.exe" --version
log "tools ready in $bin"
