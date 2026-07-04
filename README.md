# Do-Calculus Interventions in Predictive Coding Networks

Code accompanying *"Do-Calculus Interventions in Predictive Coding Networks: Formal
Correspondence, Empirical Validation, and Architectural Prescriptions for Clinical
Causal AI."*

This repo implements:

- **Proposition 1** — a formal correspondence between PC clamping (with bottom-up
  error suppression) and Pearl's `do(·)` operator in linear-Gaussian SCMs, validated
  on a 3-node toy SCM (Claim 1).
- **PCNetworkV2 / PCNetworkV3** — predictive coding architectures with a shared
  latent that decodes to covariates and outcome, evaluated against an MLP baseline
  on IHDP and MIMIC-IV semi-synthetic benchmarks (Claims 2, 3, 6).
- **Ablations** over confounding strength, interventional training fraction, and
  latent dimension (Claim 4).
- **Topology robustness** across Chain / Mediator / Collider / Fork SCMs (Claim 5).

> **Note on reproducibility.** The paper's exact numbers depend on the original
> author's random seeds, IHDP counterfactual reconstruction, and a credentialed
> MIMIC-IV Demo download. This repo reproduces the *methodology* (Equations 1–16)
> faithfully and will produce results in the same ballpark, but bit-exact
> reproduction of every table value is not guaranteed unless you use the same data
> pull and seeds. Where the paper is ambiguous, choices are documented inline in
> the code as `# NOTE:` comments.

## Repo layout

```
do-calculus-pc-networks/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── .github/
│   └── workflows/
│       └── tests.yml      # CI: runs pytest + a fast smoke test on every push/PR
├── src/
│   ├── __init__.py
│   ├── scm.py             # Toy SCM (Eq. 9) + synthetic confounded SCM topologies (Eq. 11)
│   ├── pc_networks.py     # LinearPCNetwork (Prop. 1), PCNetworkV2, PCNetworkV3 (Eq. 1-8)
│   ├── mlp_baseline.py    # MLP baseline (§2.3.3)
│   ├── data_ihdp.py       # IHDP covariates + Hill (2011) counterfactual reconstruction (Eq. 10)
│   ├── data_mimic.py      # MIMIC-IV Demo loader + counterfactual DGP (Eq. 12)
│   └── metrics.py         # MAE, gap, ATE error, bootstrap CI, permutation tests (Eq. 13-16, §2.6)
├── experiments/
│   ├── 01_toy_scm_validation.py     # Claim 1 / Figure 1 / Table 2-3
│   ├── 02_ihdp_mlp_baseline.py      # Claim 2 / Figure 2 / Table 4 (MLP rows)
│   ├── 03_ihdp_pc_vs_mlp.py         # Claim 3 / Figure 3 / Table 4 (PC rows)
│   ├── 04_ablations.py              # Claim 4 / Figure 4 / Table 5
│   ├── 05_topology_robustness.py    # Claim 5 / Figure 5 / Table 6
│   ├── 06_mimic_iv.py               # Claim 6 / Figure 6 / Table 7
│   └── generate_report.py           # Aggregates all results/*.json into results/REPORT.md
├── scripts/
│   └── run_all_experiments.sh       # Runs 01-06 + generate_report.py in order
├── tests/
│   └── test_basic.py
├── data/
│   └── README.md                    # MIMIC-IV Demo download instructions
└── results/                          # experiment outputs land here (gitignored except .gitkeep)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

- **IHDP**: this repo reconstructs covariates and counterfactual outcomes directly
  from the Hill (2011) NPCI Simulation 1 equations (Eq. 10, `src/data_ihdp.py`),
  matching the paper's approach in §2.4.2 / Limitation L2 (public IHDP mirrors are
  subject to URL rot). No external download required — `data_ihdp.load_ihdp()`
  generates the n=747, 25-covariate dataset synthetically but with the correct
  moments, and reconstructs `μ0, μ1` per Eq. 10.
- **MIMIC-IV Demo**: real ICU covariate structure requires a free PhysioNet
  credentialed/demo download (`https://physionet.org/content/mimic-iv-demo/`).
  Place the extracted CSVs under `data/mimic-iv-demo/` and set
  `MIMIC_DEMO_PATH` accordingly, or run with `--synthetic` to use a structurally
  matched synthetic fallback (11 covariates, same DGP, Eq. 12) so the pipeline
  runs end-to-end without PhysioNet access.

## Running experiments

Each script is self-contained and writes its results to `results/*.json`.
Run them all at once (recommended) and get a consolidated markdown report:

```bash
bash scripts/run_all_experiments.sh                          # synthetic MIMIC-IV fallback
bash scripts/run_all_experiments.sh --mimic-path data/mimic-iv-demo   # real MIMIC-IV Demo
```

This produces `results/REPORT.md`, formatted to mirror the paper's Tables 2–7.

Or run scripts individually:

```bash
python experiments/01_toy_scm_validation.py
python experiments/02_ihdp_mlp_baseline.py
python experiments/03_ihdp_pc_vs_mlp.py
python experiments/04_ablations.py
python experiments/05_topology_robustness.py
python experiments/06_mimic_iv.py --synthetic   # omit --synthetic if you have MIMIC-IV Demo
python experiments/generate_report.py            # aggregates whatever has been run so far
```

## Tests

```bash
pytest tests/ -v
```

## Citation

If you use this code, please cite the paper (see manuscript front matter for full
author/affiliation details once finalized).

## License

MIT — see `LICENSE`.
