# Scaling & Emergence Evals (SSE)

A framework for evaluating emergent capabilities in language models across different scales. SSE provides tools to run standardized evaluation tasks, compute metrics, and analyze scaling curves to identify phase transitions and emergent behaviors.

## Project Narrative

This project investigates the claim that language models exhibit "emergent abilities" — capabilities that appear suddenly at a certain scale. Following [Schaeffer et al. (2023)](https://arxiv.org/abs/2304.15004), we show that apparent emergence is largely an artifact of the evaluation metric: when measured with discontinuous metrics (exact-match accuracy), smooth capability improvement appears as a sudden phase transition. When measured with continuous metrics (log-probability), the same capabilities show predictable, gradual scaling.

We evaluate the [Pythia model family](https://github.com/EleutherAI/pythia) (70M–2.8B parameters) across arithmetic, factual recall, multiple-choice, and in-context learning tasks. Results and analysis are documented in [WRITEUP.md](WRITEUP.md).

## Setup

```bash
uv sync
```

## Usage

```bash
make test    # run tests
make lint    # run linter
make format  # auto-format code
make figures # regenerate all plots from results/
make writeup # convert WRITEUP.md to PDF (requires pandoc) or open markdown
```

## Running Evaluations

```bash
# Full sweep (all models × all tasks)
uv run python -m sse.runners.sweep --config configs/main_sweep.yaml

# Dry run to see the matrix
uv run python -m sse.runners.sweep --config configs/main_sweep.yaml --dry-run

# Single evaluation
uv run python -m sse.runners.run_eval --model 70m --task arithmetic_2digit --n 100
```

## Results

After running a sweep, generate publication figures and the writeup:

```bash
make figures  # outputs to results/figures/
make writeup  # generates WRITEUP.pdf
```

See [WRITEUP.md](WRITEUP.md) for the full analysis template and findings.
