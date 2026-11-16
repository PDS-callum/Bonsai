<h1 align="center">
  <img src="assets/logo.png" alt="Bonsai" width="240">
</h1>

<p align="center"><em>Small models, grown slowly on small hardware.</em></p>

<p align="center">
  <a href="https://github.com/PDS-callum/Bonsai/actions/workflows/ci.yml"><img src="https://github.com/PDS-callum/Bonsai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

Bonsai is a framework for cultivating small language models for small,
CPU-bound hardware. You start from an existing open model (the **seed**),
**prune** it to fit a hardware budget (the **pot**), shape it towards a task,
**repot** it into a format the device can run, and **measure** how it grows.
Every cut is written in the **journal**, so any tree can be traced back to its
seed and the exact steps that shaped it.

The main pot is a Raspberry Pi. A PC acts as the potting bench, where the heavy
work happens.

> **Status:** Phase 0 (Groundwork). The Python package, tooling and CI are set
> up; the cultivation tools themselves are not built yet. This README sets out
> the idea, the first phase of work (**Phase 1: Nursery**) and the decisions
> made so far. Results will be added as they come in, including the things that
> don't work.

---

## Contents

- [Why "Bonsai"](#why-bonsai)
- [The language of the garden](#the-language-of-the-garden)
- [The core idea: bytes per token](#the-core-idea-bytes-per-token)
- [The cultivation cycle](#the-cultivation-cycle)
- [The garden: hardware](#the-garden-hardware)
- [Phase 1: Nursery](#phase-1-nursery)
- [The journal](#the-journal)
- [Planned repository layout](#planned-repository-layout)
- [Engineering conventions](#engineering-conventions)
- [Development](#development)
- [Roadmap: the seasons ahead](#roadmap-the-seasons-ahead)
- [Decisions log](#decisions-log)
- [Open questions](#open-questions)
- [Prior art](#prior-art)
- [Results](#results)
- [License](#license)

---

## Why "Bonsai"

A bonsai is an ordinary tree, kept small by careful pruning and shaped with
patience until it thrives in a small pot. It isn't a different species; it's a
normal tree that has been cultivated with intent.

Bonsai treats language models the same way. Rather than training tiny models
from scratch, it takes capable open models and shapes them to live well on
small hardware: cutting what isn't needed, healing what remains, and keeping a
record of every cut.

---

## The language of the garden

The framework's vocabulary follows the metaphor, so the code and the docs read
the same way.

| Term | Meaning |
|---|---|
| **Seed** | The starting model, e.g. Qwen 0.5B from Hugging Face. |
| **Tree** | Any model in the framework: a seed or anything grown from one. |
| **Pot** | A hardware budget: RAM, storage, memory bandwidth and a target speed. |
| **Prune** | Remove structure (layers, attention heads, MLP width, experts) to fit the pot. |
| **Wire** | Retrain a pruned tree to heal it and shape it, e.g. by distillation from its seed. *(Phase 2)* |
| **Graft** | Attach new behaviour, e.g. LoRA adapters for a task. *(later)* |
| **Repot** | Quantise and convert a tree into the format the pot runs (GGUF for llama.cpp). |
| **Measure** | Score a tree on quality, speed, memory and heat. |
| **Journal** | The record of every tree, its parent, what was done to it and how it measured. |
| **Season** | One full cycle of prune → (wire) → repot → measure. |
| **Nursery** | Phase 1: raising the first trees and the tools to tend them. |

---

## The core idea: bytes per token

On a CPU, generating each token means reading every active weight from memory
once. Speed is set mostly by **how many bytes are read per token**, not by the
parameter count on the label:

```
tokens per second  ≲  memory bandwidth  ÷  bytes of active weights
```

A Raspberry Pi 5 has roughly 10–17 GB/s of memory bandwidth. A 1B-parameter
model at 4-bit is about 0.6 GB, so its ceiling is somewhere around 15–25 tokens
per second, before any other overhead.

This is what makes the pot concrete. Every pot has a bytes-per-token budget, and
every pruning or quantisation decision is judged by what it buys within that
budget. It also explains why storage-streaming tricks that let enormous models
run *at all* (reading weights from an SSD as needed) don't help a Pi run
anything *well*: a Pi's NVMe link is around 20× slower than its RAM. For daily
use, the tree has to fit in the pot.

---

## The cultivation cycle

```
   seed ──► prune ──► wire ──► repot ──► measure ──► journal
              ▲      (Phase 2)                          │
              └────────────── next season ◄─────────────┘
```

1. **Plant** a seed: register a model and its source checksum.
2. **Prune** it to fit the pot.
3. **Wire** it to recover quality and shape it for a task. *(Phase 2 onwards)*
4. **Repot** it: quantise, convert to GGUF and check it against its reference.
5. **Measure** it on the potting bench (quality) and in the pot (speed, memory,
   heat).
6. **Journal** everything, then decide what to cut next season.

---

## The garden: hardware

No extra hardware: just a PC and a Raspberry Pi on the same home network. The
Pi stays indoors in the office.

| Machine | Role |
|---|---|
| **Raspberry Pi: the pot** | Runs repotted trees with llama.cpp on CPU only. Measures speed, peak RAM, temperature, clock speed and throttling. |
| **PC: the potting bench** | Plants, prunes and repots trees. Measures quality (accuracy does not depend on the hardware for the same GGUF and settings, so it is cheaper to measure here). Sends trees to the Pi, collects measurements and keeps the journal. |

Details to record once confirmed: Pi model and RAM, cooling, OS version,
storage (SD card or NVMe), and the PC's CPU and GPU (if any).

---

## Phase 1: Nursery

**Goal:** one complete season using cheap operations only, plus the baseline
every later tree has to beat. No training yet.

### In scope

1. **Profile the pot.** Measure the Pi's real memory bandwidth, RAM and storage
   once, and store them as a pot profile. This gives each tree a speed ceiling.
2. **Measure.** Repot a tree, deploy it to the Pi's `llama-server`, and record
   prompt and generation tokens per second, time to first token, peak RAM,
   temperature and throttle flags. Measure quality on the PC, with a periodic
   cross-check on the Pi.
3. **Keep the journal.** A SQLite record of every tree: parent, operations,
   file checksums, configuration and measurements.
4. **Prune by depth.** Remove whole transformer layers from a Hugging Face
   model, choosing them with a simple importance score (how much each layer
   changes its input). Removing whole layers keeps every remaining layer the
   same shape, so the result still converts to standard GGUF.
5. **Repot with a reference check.** Convert to GGUF and quantise. Before
   measuring, check the unquantised GGUF against the pruned Hugging Face model
   on fixed prompts (matching next-token predictions within a tolerance), so a
   broken conversion is caught before it is measured.
6. **Grow the baseline.** Measure off-the-shelf seeds (e.g. Qwen and Llama 3.2,
   0.5B–3B) at several quantisation levels (Q2_K to Q8_0). This is the
   **frontier**: the best quality available at each speed without Bonsai.

### Measurements

- **Quality:** perplexity on held-out text (fast, used to guide pruning), plus
  task accuracy on fixed subsets of GSM8K (maths reasoning) and ARC (science
  knowledge), with confidence intervals.
- **Speed:** prompt and generation tokens per second, time to first token,
  time per answer.
- **Memory:** file size and peak RAM on the Pi.
- **Heat:** CPU temperature, clock speed and throttle flags over time.
- **Provenance:** seed checksum, every operation, llama.cpp commit, runtime
  settings, prompt template and random seed.

### Deliverables

- A command-line tool covering the season:

  ```
  bonsai profile             # measure the pot
  bonsai plant <model>       # register a seed
  bonsai prune <tree> ...    # cut a new tree from an existing one
  bonsai repot <tree> ...    # quantise, convert and check
  bonsai measure <tree>      # quality on the bench, speed in the pot
  bonsai journal             # browse trees and their lineage
  ```

  (Names and options are a sketch and will change.)

- **The first frontier plot:** quality against tokens per second on the Pi,
  with the off-the-shelf frontier and the pruned trees on the same axes.

### What we expect to learn

Pruning without healing will almost certainly push trees below the frontier,
possibly sharply after a few layers. That is still a useful result: it measures
how much quality wiring (Phase 2) has to recover, and which layers can be cut
cheaply.

### Not in Phase 1

Wiring (healing and distillation), grafting, task specialisation, width and
expert pruning, a custom inference engine, and growing models upward.

### Done when

- [ ] A pot profile for the Pi exists
- [ ] The off-the-shelf frontier is measured for at least two seed families
- [ ] At least one seed has been depth-pruned at several cut sizes, repotted,
      checked and measured
- [ ] Every tree in the plot can be traced through the journal to its seed
- [ ] The frontier plot and a short write-up of what was learned are in `results/`

---

## The journal

Every tree is a record. A first sketch:

| Field | Example |
|---|---|
| `id` | `qwen2.5-0.5b/p2-l3` |
| `parent` | `qwen2.5-0.5b` |
| `operation` | `prune.depth` with `{layers: [17, 18, 20]}` |
| `artifact` | path and SHA-256 of the weights or GGUF |
| `pot` | the pot profile it was measured in |
| `measurements` | quality, speed, memory and heat results, linked by run |
| `environment` | llama.cpp commit, Python dependency lock hash, config hash |

Large files (weights and GGUFs) live outside git; the journal stores their
checksums so results can be reproduced.

---

## Planned repository layout

Not yet created; a guide for when code is added.

```
Bonsai/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── assets/              # Logo and images
├── configs/
│   ├── pots/            # Hardware profiles
│   └── seasons/         # One YAML file per season or experiment
├── prompts/             # Versioned prompt templates
├── src/bonsai/
│   ├── cli.py           # The `bonsai` command
│   ├── journal.py       # SQLite lineage store
│   ├── pot.py           # Hardware profiling and telemetry on the Pi
│   ├── prune/           # Pruning operations (depth first)
│   ├── repot.py         # GGUF conversion, quantisation, reference checks
│   └── measure/         # Quality on the bench, speed and heat in the pot
├── scripts/             # Setup helpers (e.g. building llama.cpp on the Pi)
├── analysis/            # Notebooks and plotting scripts
├── results/             # Exported tables, figures and write-ups
└── tests/
```

---

## Engineering conventions

- Python 3.11+, managed with **uv**.
- **ruff** for linting and formatting, **mypy** (strict) for type checks,
  **pytest** for tests, **ShellCheck** for shell scripts.
- **pre-commit** hooks for the above; **GitHub Actions** CI on every push and
  pull request, on Python 3.11 and 3.13.
- **Conventional commits** (`feat:`, `fix:`, `docs:`, `chore:` ...).
- **Everything enters `main` through a branch and a pull request.** No direct
  commits to `main`.
- Large files (models, GGUFs, databases) stay out of git; see `.gitignore`.

---

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create the environment and install everything
uv run pre-commit install    # run the checks on every commit
uv run bonsai --version
```

The same checks CI runs:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

---

## Roadmap: the seasons ahead

**Phase 0: Groundwork**
- [ ] Record Pi and PC hardware details in this README
- [ ] Build llama.cpp on the Pi; run one 1B model by hand
- [x] Set up uv, ruff, mypy, pytest, pre-commit and CI

**Phase 1: Nursery** *(see above)*
- [ ] Pot profile, measuring, journal
- [ ] Off-the-shelf frontier
- [ ] Depth pruning, repotting and reference checks
- [ ] First frontier plot and write-up

**Phase 2: Wiring**
- [ ] Heal pruned trees by distillation from their seed
- [ ] Width pruning (attention heads, MLP channels), kept GGUF-compatible
- [ ] Do healed trees reach the frontier, or beat it?

**Phase 3: Shaping**
- [ ] Task-specific cultivation: prune and wire a tree for one job (e.g. maths
      or JSON extraction)
- [ ] Grammar-constrained output as part of the shape
- [ ] Equal-time comparisons: one large answer vs several small answers with a
      majority vote

**Later: the wider garden**
- [ ] Expert pruning on small mixture-of-experts seeds, keeping only the
      experts a task uses
- [ ] Streaming experts from NVMe on the Pi, as an experiment
- [ ] A small dependency-free inference engine, freeing pruning from GGUF's
      same-shape-per-layer limit
- [ ] Growing models upward rather than cutting them down
- [ ] Long sustained runs and throttling analysis

---

## Decisions log

| Decision | Reason |
|---|---|
| A framework for cultivating models, not only a benchmark | Measuring alone answers "what exists"; cultivating asks "what can we make". The benchmark becomes the baseline. |
| Bytes per token as the central budget | On CPUs, memory bandwidth sets the speed ceiling. |
| CPU only, Pi + PC, no extra hardware | Keep it focused on what small hardware can do. |
| llama.cpp and GGUF as the first runtime | Widely used, runs well on ARM, wide range of quantisation levels. |
| Quality measured on the PC, speed and heat on the Pi | Accuracy doesn't depend on the hardware for the same GGUF and settings; Pi time is the scarce resource. |
| Depth pruning first | Cheap, well studied, and keeps models GGUF-compatible. |
| No training in Phase 1 | Training needs a GPU decision; pruning alone establishes the loop and the gap healing must close. |
| SQLite for the journal | One file, easy to query, no server. |
| uv, ruff and strict mypy | Fast, reproducible environments from a lock file; one tool for linting and formatting; types checked everywhere. |
| Typer for the CLI | Commands defined from type hints, with little boilerplate. |
| Everything into `main` via pull request | Keeps history reviewable. |

---

## Open questions

- Which Pi model and how much RAM? SD card or NVMe? This sets the pot.
- Does the PC have a GPU, and how much VRAM? This decides whether Phase 2
  wiring can run locally.
- Which task should the first specialised tree be shaped for?
- Which importance score works best for choosing layers to cut on small models?
- How many repeats per configuration are enough for stable confidence
  intervals?
- Is active cooling needed for fair sustained measurements?

---

## Prior art

Bonsai builds on published work rather than inventing new techniques:

- [llama.cpp](https://github.com/ggml-org/llama.cpp): the CPU runtime and GGUF
  format.
- Layer removal: ShortGPT; *The Unreasonable Ineffectiveness of the Deeper
  Layers*.
- Structured pruning: LLM-Pruner, SliceGPT, Sheared-LLaMA.
- Prune then distill: NVIDIA's Minitron.
- Minimal inference engines: [llama2.c](https://github.com/karpathy/llama2.c).

---

## Results

None yet.

---

## License

Bonsai's code is licensed under the [Apache License 2.0](LICENSE).

The license covers this repository's code only, not the models it works on.
Every seed keeps its own license, and so does every tree grown from it: Qwen2.5
models are Apache-2.0, while Llama 3.2 models are under Meta's Llama 3.2
Community License. Check a seed's terms before sharing what you grow from it.
