# Scripts

Setup helpers for the two machines in the garden. Both use the llama.cpp
version pinned in [`configs/llama-cpp.env`](../configs/llama-cpp.env) (ref,
commit and numbered build), so the potting bench and the pot always run
identical code.

## The pot: Raspberry Pi

Build llama.cpp on the Pi, started from the PC:

```bash
BONSAI_PI=user@host scripts/pi/remote-build.sh
```

This copies the scripts to `~/bonsai/scripts` on the Pi and starts the build as
a detached job, so it carries on if the SSH connection drops. Follow it with:

```bash
ssh user@host tail -f bonsai/jobs/build-llama-cpp/log.txt
```

When it finishes, `bonsai/jobs/build-llama-cpp/exit_code` holds the exit status,
and `bonsai/llama.cpp/build/bin/build-info.json` records the commit, compiler and
flags. A full build takes about 8–9 minutes on a Pi 5 with two compile jobs.

| Script | Runs on | Purpose |
|---|---|---|
| [`pi/remote-build.sh`](pi/remote-build.sh) | PC | Copy the Pi scripts over and start a headless build. |
| [`pi/build-llama-cpp.sh`](pi/build-llama-cpp.sh) | Pi | Build `llama-cli`, `llama-bench` and `llama-server` with native ARM optimisations. Installs cmake from PyPI if needed, so no root access is required. |
| [`pi/run-detached.sh`](pi/run-detached.sh) | Pi | Run any command in a named tmux session, logging to a job directory and recording its exit code. |

### Profiling the pot

`profile-pot.sh` measures the facts behind a pot profile: memory bandwidth at
each thread count, free memory, storage, CPU features and thermals. Copy the Pi
scripts over (as `remote-build.sh` does), then start it detached:

```bash
ssh user@host 'bonsai/scripts/pi/run-detached.sh profile-pot     bonsai/jobs/profile-pot -- bonsai/scripts/pi/profile-pot.sh results'
```

It takes under a minute and uses about 768 MiB of memory. Results land in
`bonsai/jobs/profile-pot/results/`: `membw.jsonl` (every bandwidth sample),
`telemetry.csv` (temperature, clock and throttling each second) and
`facts-before.json` / `facts-after.json`. Copy them to `results/pots/<pot>/<date>/`
and update the pot's profile in [`configs/pots/`](../configs/pots).

| Script | Runs on | Purpose |
|---|---|---|
| [`pi/profile-pot.sh`](pi/profile-pot.sh) | Pi | Run the pot measurements below, sampling thermals throughout. |
| [`pi/membw.c`](pi/membw.c) | Pi | Read, copy and triad bandwidth over arrays far larger than the caches; prints every sample. |
| [`pi/pot_facts.py`](pi/pot_facts.py) | Pi | JSON snapshot of hardware, memory, storage and thermals, standard library only. |

## The potting bench: PC

```bash
scripts/pc/fetch-llama-cpp.sh
```

Clones the llama.cpp source at the pinned ref (for `convert_hf_to_gguf.py`) and,
on Windows x64, downloads the official prebuilt CPU binaries for the same commit
(including `llama-quantize`), checking their SHA-256 checksum. Everything goes
under `~/bonsai` (override with `BONSAI_HOME`).
