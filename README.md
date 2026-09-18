# Causal IV Proxy Time-Series Experiment

A Python research prototype for causal discovery in synthetic time-series data using instrumental-variable (IV) and proxy-variable ideas. The project generates a simulated dataset, estimates candidate causal relationships, and exports discovered edges, graph data, and evaluation metrics.

> **Status:** Research / experimental code. Results depend on the synthetic data-generating process and the causal assumptions built into the method. They are not, by themselves, evidence of causal effects in real-world data.

## Overview

The project is contained in `causal_iv_proxy_ts/` and centers on a single Python program, `causal_iv_proxy_ts.py`. That script implements the experimental pipeline, while `requirements.txt` declares its Python dependencies. Generated artifacts are stored in `causal_iv_proxy_ts/results/`.

The experiment is designed to:

1. Generate synthetic time-series observations.
2. Apply an IV/proxy-aware causal-discovery procedure.
3. Produce a list of discovered directed edges.
4. Export a graph edge list for downstream visualization or analysis.
5. Calculate and save evaluation metrics.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `causal_iv_proxy_ts/causal_iv_proxy_ts.py` | Main implementation and experiment entry point. |
| `causal_iv_proxy_ts/requirements.txt` | Python dependencies needed to run the experiment. |
| `causal_iv_proxy_ts/results/synthetic_data.csv` | Saved synthetic time-series dataset from an experiment run. |
| `causal_iv_proxy_ts/results/discovered_edges.csv` | Discovered causal-edge candidates and associated result information. |
| `causal_iv_proxy_ts/results/graph.edgelist` | Edge-list representation of the discovered graph. |
| `causal_iv_proxy_ts/results/metrics.json` | Evaluation metrics saved from the experiment. |

## Requirements

- Python 3.
- The packages listed in `causal_iv_proxy_ts/requirements.txt`.
- A virtual environment is recommended to isolate the experiment dependencies.

## Setup

Clone the repository, create an environment, and install dependencies:

```bash
git clone https://github.com/Verlopat/13.git
cd 13/causal_iv_proxy_ts

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the experiment

Run the main script from the project directory:

```bash
python3 causal_iv_proxy_ts.py
```

The script writes its outputs to the `results/` directory. Existing artifacts may be overwritten, so copy or version results before rerunning an experiment whose output you need to retain.

## Output artifacts

| Artifact | Use |
| --- | --- |
| `synthetic_data.csv` | Input time-series data generated for the run. |
| `discovered_edges.csv` | Candidate causal relationships returned by the method. |
| `graph.edgelist` | Graph structure in edge-list form for external tools. |
| `metrics.json` | Quantitative evaluation results for the run. |

The generated files should be read together: edge discovery indicates the relationships inferred by the method, while `metrics.json` summarizes performance against the experiment’s own reference assumptions.

## Reproducibility

For meaningful comparisons across runs, record:

- The repository commit and the version of Python.
- Exact package versions from the installed environment.
- The parameters and random seed used by `causal_iv_proxy_ts.py`.
- The synthetic data file and any ground-truth graph used for scoring.
- The command and environment used to produce each result artifact.

If the implementation currently does not expose a seed or configuration interface, add one before conducting a systematic study.

## Interpretation and limits

- IV- and proxy-based causal approaches require explicit assumptions about relevance, exclusion, independence, and how proxies relate to latent or confounding variables.
- Synthetic-data performance shows how this implementation behaves under its selected model; it does not establish performance under every possible data-generating process.
- Discovered graph edges are model outputs, not automatically confirmed causal relationships.
- Before applying this approach to observational or operational data, assess identification conditions with subject-matter and statistical expertise.

## Repository layout

```text
.
├── README.md
└── causal_iv_proxy_ts/
    ├── causal_iv_proxy_ts.py       # Main experiment implementation
    ├── requirements.txt            # Python dependencies
    └── results/
        ├── synthetic_data.csv      # Generated experiment data
        ├── discovered_edges.csv    # Inferred edge candidates
        ├── graph.edgelist          # Graph edge-list output
        └── metrics.json            # Evaluation metrics
```

## Suggested improvements

1. Add a configuration file or command-line options for sample size, lag depth, thresholds, random seeds, and output paths.
2. Document the synthetic data-generating process, variables, lag conventions, true graph, and IV/proxy assumptions.
3. Add unit tests for the identification logic, edge scoring, and output serialization.
4. Add baseline methods and report precision, recall, false-discovery behavior, runtime, and sensitivity to assumption violations.
5. Provide a notebook or script to visualize `graph.edgelist`, `discovered_edges.csv`, and `metrics.json`.
6. Pin dependency versions for reproducible results.
7. Add a license and contribution guidelines.

## License

No license file is currently included. Add an explicit license before distributing or accepting external contributions.
