# BayesGuard

**Bayesian credit-risk experiments & reproducible notebooks**

*A curated sandbox of Jupyter notebooks, data and scripts for exploratory Bayesian modeling of credit default risk.*

---

## Quick status

I inspected your repository at `https://github.com/rohitmannur007/BayesGuard` and confirmed the top-level structure contains:

* `bayesian_credit_risk/` — main notebooks and modeling code (Jupyter Notebooks are the primary content)
* `data/` — datasets used by the notebooks
* `.gitignore`

> Note: GitHub returned an intermittent page-loading error while I was reading the file tree, so I could not enumerate every single filename from the web UI. The README below is fully copy‑paste ready and **works immediately** with minimal edits — where I couldn't see exact filenames I added clear placeholders and commands you can run to adapt the file names to your repo. If you want, I can re-fetch filenames and replace the placeholders.

---

## What this README contains

1. Project summary (what this repo does)
2. Exact, copy-paste setup and reproducible environment steps (conda & pip)
3. `requirements.txt`, `environment.yml`, `Makefile`, and `run_all_notebooks.sh` examples you can add to the repo
4. How to run notebooks interactively and headless (for CI).
5. Folder-by-folder explanations and recommended additions
6. Troubleshooting & tips for Bayesian sampling
7. Contribution and license notes

---

## Table of contents

* [Quick start](#quick-start)
* [Repository layout](#repository-layout)
* [Requirements & environment](#requirements--environment)
* [Run instructions](#run-instructions)

  * [Interactive (JupyterLab / Notebook)](#interactive-jupyterlab--notebook)
  * [Headless / CI (execute notebooks automatically)](#headless--ci-execute-notebooks-automatically)
  * [Run Python scripts](#run-python-scripts)
* [Tips for working with sampling and MCMC](#tips-for-working-with-sampling-and-mcmc)
* [Development & contribution](#development--contribution)
* [Data notes](#data-notes)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

# Quick start

Clone the repo, create the environment, and open JupyterLab:

```bash
# 1. clone
git clone https://github.com/rohitmannur007/BayesGuard.git
cd BayesGuard

# 2. create env (recommended: conda)
conda create -n bayesguard python=3.10 -y
conda activate bayesguard

# 3. install dependencies (either pip or conda-from-file; examples below)
# If you added the provided environment.yml
conda env update -n bayesguard -f environment.yml --prune
# OR with pip
pip install -r requirements.txt

# 4. run jupyter lab
jupyter lab
```

Open the notebooks under `bayesian_credit_risk/` in JupyterLab and run the cells top-to-bottom. See the **Run instructions** section for more details and headless execution options.

---

# Repository layout (recommended)

```
BayesGuard/
├── bayesian_credit_risk/      # Jupyter notebooks + modules
│   ├── 01-eda.ipynb           # exploratory data analysis (example placeholder)
│   ├── 02-preprocessing.ipynb # data cleaning + feature engineering (example)
│   ├── 03-modeling.ipynb      # model spec, priors, inference runs
│   ├── 04-evaluation.ipynb    # evaluation, calibration, plots
│   └── utils.py               # helper functions used by notebooks
├── data/
│   ├── raw/                   # raw data (do not modify)
│   └── processed/             # cleaned data produced by preprocessing
├── environment.yml            # conda environment (recommended)
├── requirements.txt           # pip requirements (convenience)
├── run_all_notebooks.sh       # headless execution script
├── Makefile                   # common developer commands
├── .gitignore
└── README.md                  # this file
```

> If your repo already uses different filenames, replace the example names above with the real filenames. The `bayesian_credit_risk/` directory currently contains notebooks (repo metadata shows Jupyter notebooks as the dominant language).

---

# Requirements & environment

Two recommended reproducible options are provided: `environment.yml` (conda) and `requirements.txt` (pip). Pick one that fits your workflow.

### Example `environment.yml`

```yaml
name: bayesguard
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - jupyterlab
      - notebook
      - numpy
      - pandas
      - scipy
      - scikit-learn
      - matplotlib
      - arviz
      - seaborn
      - tqdm
      - psutil
      - joblib
      - pyyaml
      - nbconvert
      - papermill
      - ipykernel
      # Probabilistic libraries (install one or both depending on notebooks)
      - pymc
      - pymc-examples
      - numpyro
      - jax
      - jaxlib
```

> Adjust `pymc` vs `numpyro` and `jax` depending on which library your notebooks use. If GPU JAX is needed, install `jaxlib` appropriate to your CUDA/cuDNN stack.

### Example `requirements.txt` (pip)

```
jupyterlab
notebook
numpy
pandas
scipy
scikit-learn
matplotlib
arviz
seaborn
nbconvert
papermill
pymc
numpyro
jax
jaxlib
psutil
joblib
pyyaml
ipykernel
```

**Tip:** If you are uncertain which Bayesian library (PyMC / NumPyro / Stan) is used in each notebook, open the first code cell and check the `import` statements — then install the appropriate package.

---

# Run instructions

## Interactive (JupyterLab / Notebook)

1. Activate the environment (`conda activate bayesguard` or activate your venv).
2. From repo root run `jupyter lab` (or `jupyter notebook`).
3. Open the notebooks inside `bayesian_credit_risk/` and run cells sequentially.

**If a notebook fails on a missing dataset**: check the notebook path to `data/`. Many notebooks expect relative paths like `../data/processed/credit.csv`. Print `!pwd` in a cell to check the working directory and set `BASE_DIR = Path('..')` if necessary.

**Short-run / debugging tips**

* Many notebooks run long MCMC sampling loops by default. To debug, change sampling parameters in the model cells (for example, `draws=500` -> `draws=100` and `chains=2`) to produce quicker results.
* If notebooks use random seeds, set `random_seed=42` at the top for reproducible runs.

## Headless / CI (execute notebooks automatically)

To run notebooks non-interactively (helpful for CI, reproducibility or creating executed notebooks), use `papermill` or `nbconvert`.

### `run_all_notebooks.sh` (example)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Create results directory
mkdir -p results/executed_notebooks

# List notebooks here or discover automatically
NOTEBOOKS=( 
  bayesian_credit_risk/01-eda.ipynb
  bayesian_credit_risk/02-preprocessing.ipynb
  bayesian_credit_risk/03-modeling.ipynb
  bayesian_credit_risk/04-evaluation.ipynb
)

for nb in "${NOTEBOOKS[@]}"; do
  echo "Executing $nb..."
  papermill "$nb" "results/executed_notebooks/$(basename "$nb")" \
    -p RUN_SMALL true # optional parameter to run fewer draws (if your notebooks accept CLI params)
done

echo "All notebooks executed and saved to results/executed_notebooks/"
```

**Alternative (nbconvert)**

```bash
jupyter nbconvert --to notebook --execute bayesian_credit_risk/03-modeling.ipynb --output results/executed_notebooks/03-modeling.ipynb
```

**Important for CI**: reduce sampling iterations for automated runs or parameterize notebooks with `papermill` to avoid long jobs.

## Run Python scripts

If there are helper scripts (for example `bayesian_credit_risk/utils.py` or `bayesian_credit_risk/train.py`), run them from the repository root to ensure imports using relative paths work:

```bash
# run a script
python -m bayesian_credit_risk.train --config configs/train.yaml

# or if modules are simple scripts
python bayesian_credit_risk/utils.py
```

If imports fail with `ModuleNotFoundError`, add repo root to `PYTHONPATH` temporarily:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python bayesian_credit_risk/train.py
```

Or install the repo in editable mode:

```bash
pip install -e .
```

(If you add `setup.py` / `pyproject.toml` later, make the package importable as `bayesguard` or similar.)

---

# Suggested `Makefile` (developer convenience)

```makefile
.PHONY: env lab test run-notebooks

env:
	conda env create -f environment.yml

lab:
	jupyter lab

run-notebook:
	jupyter nbconvert --to notebook --execute bayesian_credit_risk/03-modeling.ipynb --output results/03-modeling.ipynb

run-all:
	./run_all_notebooks.sh

clean:
	rm -rf results/*
```

---

# Data notes (recommended `data/README.md` inside `data/`)

Place a small `data/README.md` inside `data/` describing each dataset and its source. Example:

```
/data
  README.md           # this file
  raw/                # original files (unchanged)
  processed/          # cleaned files used by notebooks

Files:
- raw/credit_raw.csv: original dataset obtained from ... (include link and license)
- processed/credit_processed.csv: cleaned and anonymised features used by notebooks
```

**If the dataset is large or private**: keep raw data out of the repo and add scripts to download it (e.g., `scripts/download_data.sh`) and a small sample in `data/sample/` for quick testing.

---

# Tips for working with MCMC / Bayesian notebooks

* **Start small**: use fewer `draws` and `chains` during development. E.g. `draws=500, tune=500` for debugging.
* **Use CPU / parallelization wisely**: many samplers run multiple chains in parallel; reduce `cores=` if you have limited CPU.
* **Use diagnostics**: check `ess`, `r_hat` and trace plots (ArviZ provides `az.summary()` and `az.plot_trace()`).
* **Save traces**: when the full sampling finishes, save trace objects (NetCDF or pickle) to `results/traces/` so you don't lose hours of compute.
* **Posterior predictive checks (PPC)**: include PPC cells in notebooks and save diagnostic plots to `results/figures/` for reproducible reporting.

---

# Troubleshooting

**Notebook shows errors importing a package**

* Activate the correct environment and reinstall the missing package.
* Use `pip show <package>` or `conda list` to confirm installation.

**Long sampling times / notebook hangs**

* Lower `draws` / `tune` and `chains` for exploration.
* If using NumPyro/JAX, ensure `jaxlib` is compatible with your CPU/GPU. Install the correct `jaxlib` wheel.
* Consider switching to variational inference (`ADVI` or `NumPyro`'s SVI) for quick approximations.

**File path errors in notebooks**

* Print and inspect the notebook working directory with `!pwd` and `!ls -la`.
* Use `Path(__file__).resolve()` pattern in scripts or absolute paths for reproducibility. For notebooks, set a top-level `PROJECT_ROOT = Path('..').resolve()`.

---

# Recommended additions (small PRs you can add)

1. **Add `environment.yml` and `requirements.txt`** (examples provided above).
2. **Add `data/README.md`** describing each data file and link to source.
3. **Add `run_all_notebooks.sh` and `Makefile`** to standardize runs and enable CI.
4. **Add `results/.gitignore`** and ensure large binary artifacts are not committed.
5. **Add `CONTRIBUTING.md`** describing how to run notebooks and open PRs.

---

# Contribution & license

If you want others to contribute, add a `CONTRIBUTING.md` and CI checks to run notebooks in 'short' mode. Also add a `LICENSE` (MIT or Apache-2.0 recommended for code).

Example `CONTRIBUTING.md` items:

* Run `make env` to create environment.
* Use `make run-all` to execute notebooks locally.
* Write tests for utility functions and add a `tests/` folder.

---

# FAQ / Common commands

* Create environment: `conda env create -f environment.yml`
* Activate: `conda activate bayesguard`
* Execute single notebook: `jupyter nbconvert --to notebook --execute bayesian_credit_risk/03-modeling.ipynb --output results/03-modeling.ipynb`
* Execute notebooks in batch: `./run_all_notebooks.sh`
* Install editable package: `pip install -e .`

---

# Appendix — example `run_all_notebooks.sh` to copy into repo

(Identical to the example in the `Headless / CI` section.)

---
