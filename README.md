<h1 align="center">
  <img src="assets/logo.png" alt="Bonsai" width="240">
</h1>

<p align="center"><em>Small models, grown slowly on small hardware.</em></p>

Bonsai is an ongoing research project into running small language models on
CPU, mainly on a Raspberry Pi, with a PC for heavier lifting. The aim is to find
out what these models can actually do under tight constraints, and what it
costs to get there: how much accuracy is lost to compression, which
optimisations really help on a Pi's ARM processor, and how best to spend a
fixed time budget.

> **Status:** early days. Nothing below has been run yet; this README sets out
> the plan, the decisions made so far and the open questions. Results will be
> added as they come in, including the things that don't work.

---

## Contents

- [Why "Bonsai"](#why-bonsai)
- [Research questions](#research-questions)
- [Hardware and roles](#hardware-and-roles)
- [Software stack](#software-stack)
- [Models](#models)
- [Evaluation tasks](#evaluation-tasks)
- [Experiments](#experiments)
- [What is measured](#what-is-measured)
- [Methodology and reproducibility](#methodology-and-reproducibility)
- [Planned repository layout](#planned-repository-layout)
- [Engineering conventions](#engineering-conventions)
- [Roadmap](#roadmap)
- [Decisions log](#decisions-log)
- [Open questions](#open-questions)
- [Results](#results)

---

## Why "Bonsai"

A bonsai is a small tree shaped by careful pruning and patience, grown in a
small pot. This project does the same with language models: prune and compress
them, then see what grows well on small hardware. Pruning is also a real model
compression technique.

---

## Research questions

The main question:

> **With a fixed time budget on a Raspberry Pi, what gives the best answers: a
> bigger model compressed hard, a smaller model at higher precision, or a tiny
> model run several times?**

Supporting questions:

1. **Quantisation vs accuracy.** How does accuracy change from Q8 down to Q2?
   Where does quality fall off a cliff, and is that point the same for every
   model and task?
2. **Reasoning vs knowledge.** Does compression hurt multi-step reasoning more
   than factual recall?
3. **Equal-time comparisons.** At the same seconds per question, does a larger
   model answering once beat a smaller model answering several times with a
   majority vote (self-consistency)?
4. **Inference optimisations on ARM.** Which settings and techniques actually
   help on a Pi CPU, as opposed to on GPUs: thread count, context length, KV
   cache quantisation, prompt caching and speculative decoding?
5. **Constrained output.** Does forcing output into a valid format with a
   grammar (e.g. JSON) let a small model match a bigger one on extraction
   tasks?
6. **Sustained performance.** Does the Pi throttle as it heats up over long
   runs, and how much does that change real throughput compared with a single
   best-case number?

---

## Hardware and roles

No extra hardware: just a PC and a Raspberry Pi on the same home network. The
Pi stays indoors in the office.

| Machine | Role |
|---|---|
| **Raspberry Pi** | The device under test. Runs quantised models with llama.cpp on CPU only. Logs temperature, clock speed and throttling. |
| **PC** | The orchestrator. Prepares and quantises models, runs full-precision reference results, sends prompts to the Pi over SSH or HTTP, scores answers and stores results. |

Details to record once confirmed: Pi model and RAM, cooling (heatsink or fan),
OS version, storage (SD card or SSD), and the PC's CPU and GPU (if any).

---

## Software stack

Planned, not yet installed:

- **Inference:** [llama.cpp](https://github.com/ggml-org/llama.cpp), built
  from source on the Pi. Models in GGUF format. Its built-in server
  (`llama-server`) provides an HTTP API the PC can call.
- **Quantisation:** llama.cpp's quantisation tools, run on the PC.
- **Orchestration and evaluation:** Python 3.11+, managed with uv.
- **Results storage:** SQLite, one row per (configuration, question, run).
- **Analysis and plots:** pandas, matplotlib or Plotly, in notebooks or
  scripts under `analysis/`.
- **Pi telemetry:** `vcgencmd` (`measure_temp`, `measure_clock arm`,
  `get_throttled`; `pmic_read_adc` for power readings on a Pi 5).

---

## Models

Small open-weight models from about 0.5B to 4B parameters. Candidates, to be
finalised (check each licence before use):

- Qwen (0.5B, 1.5B, 3B)
- Llama 3.2 (1B, 3B)
- Gemma (small variants)
- SmolLM (135M to 1.7B)
- Phi (mini variants)

Each model is tested at several quantisation levels (for example Q2_K, Q3_K_M,
Q4_K_M, Q5_K_M, Q6_K, Q8_0), with full-precision results on the PC as the
reference. Model files are not stored in git; record exact sources and
checksums in `configs/models.yaml`.

---

## Evaluation tasks

Fixed subsets of public benchmarks, chosen to cover different skills. The same
question IDs are used for every configuration so results are directly
comparable.

| Skill | Dataset (candidates) | Scoring |
|---|---|---|
| Multi-step maths reasoning | GSM8K | Exact match on the final number |
| Science knowledge | ARC (Easy/Challenge), SciQ | Multiple-choice accuracy |
| Structured extraction | A small extraction set (to be chosen or built) | Valid JSON rate, field-level accuracy |

Start with about 100 questions per task to keep runs short, then increase once
the harness is stable. Record dataset versions and the exact question IDs used.

---

## Experiments

| # | Experiment | Varies | Fixed |
|---|---|---|---|
| E1 | Quantisation sweep | Model × quantisation level | Prompt, settings, questions |
| E2 | Equal time budget | Model size, number of samples (majority vote) | Seconds per question |
| E3 | Inference optimisations | Threads, context length, KV cache type, prompt caching, speculative decoding (draft model) | Model, quantisation, questions |
| E4 | Constrained output | Grammar on or off | Model, quantisation, extraction task |
| E5 | Sustained running | Run length (e.g. 1 hour+) | Everything else |

E1 comes first: it builds the harness and gives the baseline every later
experiment is compared against.

---

## What is measured

For every run:

- **Quality:** accuracy (or task-specific score), with confidence intervals.
- **Speed:** prompt processing tokens per second, generation tokens per second,
  time to first token, total time per question.
- **Memory:** peak RAM use on the Pi.
- **Hardware state:** CPU temperature, clock speed and throttle flags over
  time; power on a Pi 5 if readings are reliable.
- **Configuration:** model, quantisation, llama.cpp commit, all runtime
  settings, prompt template, random seed.

Headline outputs:

- Accuracy against speed and against memory, showing which configurations are
  best on every measure at once (the Pareto frontier).
- Accuracy against quantisation level for each model and task.
- Throughput and temperature over time for sustained runs.
- Error analysis: which kinds of questions break first under compression.

---

## Methodology and reproducibility

- **Repeated runs:** each configuration runs several times; report means with
  confidence intervals (e.g. bootstrap over questions), not single numbers.
- **Fixed inputs:** the same questions, prompt templates and seeds for every
  configuration. Prompts live in versioned files, not inline in code.
- **One config per experiment:** each run is fully described by a YAML file in
  `configs/`; results record the config's hash.
- **Pinned everything:** model checksums, llama.cpp commit hash, Python
  dependencies via `uv.lock`.
- **Consistent hardware state:** note cooling setup, let the Pi cool between
  runs (or record the starting temperature), and avoid other load during runs.
- **Report negatives:** optimisations that don't help are results too.

---

## Planned repository layout

Not yet created; a guide for when code is added.

```
Bonsai/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/            # YAML experiment and model configs
├── prompts/            # Versioned prompt templates
├── src/bonsai/
│   ├── runner.py       # Sends prompts to the Pi, collects outputs and timings
│   ├── tasks/          # Dataset loading and scoring per task
│   ├── telemetry.py    # Pi temperature, clock, throttling, power
│   └── store.py        # SQLite results storage
├── scripts/            # Setup helpers (e.g. building llama.cpp on the Pi)
├── analysis/           # Notebooks and plotting scripts
├── results/            # Exported tables and figures (databases are git-ignored)
└── tests/
```

---

## Engineering conventions

- Python 3.11+, managed with **uv**.
- **ruff** for linting and formatting, **mypy** for type checks, **pytest**
  for tests.
- **pre-commit** hooks for the above; **GitHub Actions** CI running them on
  every push.
- **Conventional commits** (`feat:`, `fix:`, `docs:`, `chore:` ...).
- Large files (models, results databases) stay out of git; see `.gitignore`.

---

## Roadmap

**Phase 0: setup**
- [ ] Record Pi and PC hardware details in this README
- [ ] Build llama.cpp on the Pi; run one 1B model by hand
- [ ] Set up uv, ruff, mypy, pytest, pre-commit and CI

**Phase 1: first result (E1, small)**
- [ ] Minimal harness: PC sends a question to the Pi, gets the answer, scores
      it and stores speed, accuracy and temperature
- [ ] One 1B model at three quantisation levels on 100 GSM8K questions
- [ ] First accuracy-against-speed plot

**Phase 2: full quantisation sweep (E1)**
- [ ] More models and quantisation levels; add ARC/SciQ
- [ ] Repeated runs and confidence intervals
- [ ] Error analysis by question type

**Phase 3: the main question and optimisations (E2, E3)**
- [ ] Equal-time-budget comparison with majority voting
- [ ] Inference optimisation study

**Phase 4: constrained output and sustained running (E4, E5)**
- [ ] Grammar-constrained extraction
- [ ] Long runs with throttling analysis

**Phase 5: write-up**
- [ ] Findings section in this README (or a separate write-up), including what
      didn't work

---

## Decisions log

| Decision | Reason |
|---|---|
| CPU only, Pi + PC, no extra hardware | Keep it simple and focused on what small hardware can do. |
| llama.cpp and GGUF | The most widely used CPU inference engine; runs well on ARM; wide range of quantisation levels. |
| PC orchestrates, Pi only runs models | Keeps the Pi's resources for the model being measured. |
| Public benchmarks with fixed subsets | Comparable across configurations and with published numbers; no labelling needed. |
| SQLite for results | One file, easy to query, no server. |

---

## Open questions

- Which Pi model and how much RAM? This sets the largest usable model.
- Is active cooling needed for fair sustained-performance runs?
- Which extraction dataset to use, or build a small one?
- How to measure power reliably without extra hardware?
- How many repeats per configuration are enough for stable confidence
  intervals?

---

## Results

None yet.
