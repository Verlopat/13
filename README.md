# Causal IV Proxy Time-Series Experiment

Python research prototype for causal discovery in synthetic time-series data using instrumental-variable and proxy-variable ideas.

## Pipeline

`causal_iv_proxy_ts/causal_iv_proxy_ts.py` generates synthetic observations, applies the IV/proxy-aware procedure, exports discovered directed edges and a graph edge list, and saves evaluation metrics. Results are stored under `causal_iv_proxy_ts/results/` as `synthetic_data.csv`, `discovered_edges.csv`, `graph.edgelist`, and `metrics.json`.

## Setup and run

```bash
git clone https://github.com/Verlopat/13.git
cd 13/causal_iv_proxy_ts
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python3 causal_iv_proxy_ts.py
```

Existing results may be overwritten. Record the commit, Python/dependency versions, parameters, random seed, and generated inputs for each run.

## Interpretation

IV and proxy methods require relevance, exclusion, independence, and valid proxy assumptions. Synthetic performance describes this implementation under its chosen model; discovered edges are model outputs and should not be treated as confirmed causal relationships. Add configuration options, tests, baselines, pinned dependencies, and a results schema before systematic study.

## License

No license file is currently included.
