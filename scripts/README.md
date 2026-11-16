# Scripts

Setup and measuring scripts for the two machines in the garden. Both machines
use the llama.cpp version pinned in
[`configs/llama-cpp.env`](../configs/llama-cpp.env) (ref, commit and numbered
build), so the potting bench and the pot always run identical code.

Day to day, `bonsai job` drives the Pi scripts (`bonsai job profile`,
`bonsai job bench`); the commands below are what it runs, for use by hand.

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
ssh user@host bonsai/scripts/pi/run-detached.sh profile-pot \
    bonsai/jobs/profile-pot -- bonsai/scripts/pi/profile-pot.sh results
```

It takes under a minute and uses about 768 MiB of memory. Results land in
`bonsai/jobs/profile-pot/results/`: `membw.jsonl` (every bandwidth sample),
`telemetry.csv` (temperature, clock and throttling each second) and
`facts-before.json` / `facts-after.json`. Copy them to `results/pots/<pot>/<date>/`
and update the pot's profile in [`configs/pots/`](../configs/pots).

| Script | Runs on | Purpose |
|---|---|---|
| [`pi/profile-pot.sh`](pi/profile-pot.sh) | Pi | Run all the pot measurements, sampling thermals throughout. |
| [`pi/membw.c`](pi/membw.c) | Pi | Read, copy and triad bandwidth over arrays far larger than the caches; prints every sample. |
| [`pi/pot_facts.py`](pi/pot_facts.py) | Pi | JSON snapshot of hardware, memory, storage and thermals, standard library only. |

### Measuring trees

`bench-trees.sh` measures every GGUF in a directory: prompt and generation speed
with `llama-bench`, and each run's peak memory. It then runs a reference tree
at 1–4 threads, and once more over a long generation run to see whether heat
lowers the clock. Telemetry is sampled every second throughout.

| Script | Runs on | Purpose |
|---|---|---|
| [`pi/bench-trees.sh`](pi/bench-trees.sh) | Pi | Speed, memory and heat for each tree, plus thread and sustained runs. |
| [`pi/telemetry.sh`](pi/telemetry.sh) | Pi | CSV of temperature, clock, throttling, load and free memory, once a second. |
| [`pi/peak_rss.py`](pi/peak_rss.py) | Pi | Run a command and record its peak memory (GNU `time` isn't installed). |

## The potting bench: PC

```bash
scripts/pc/fetch-llama-cpp.sh
```

Clones the llama.cpp source at the pinned ref (for `convert_hf_to_gguf.py`) and,
on Windows x64, downloads the official prebuilt CPU binaries for the same commit
(including `llama-quantize`), checking their SHA-256 checksum. Everything goes
under `~/bonsai` (override with `BONSAI_HOME`).

### Fetching seeds

```bash
uv run python scripts/pc/fetch_gguf.py configs/seeds/qwen2.5-0.5b-instruct-gguf.toml
```

Downloads the GGUFs a seed manifest in [`configs/seeds/`](../configs/seeds)
lists, from Hugging Face at the exact revision it pins. Each file is checked
against its SHA-256 before it is moved into place under
`~/bonsai/seeds/<name>/`. Files already present and correct are skipped.
