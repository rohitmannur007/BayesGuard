# Bayesian Credit Default Risk Modeling

This project implements a Bayesian hierarchical logistic regression model for credit default prediction using the German Credit dataset. It demonstrates uncertainty quantification, model calibration, and governance for regulatory compliance.

## Setup
- Activate venv: `source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run: `jupyter notebook` and open notebooks/main.ipynb

## Key Components
- Data: Loaded from https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv
- Model: Hierarchical Bayesian logistic regression (varying intercepts by loan purpose)
- Inference: MCMC with NumPyro
- Validation: Posterior predictive checks, calibration plots, expected loss thresholding

For Mac M2: Runs on CPU; MCMC completes in <1 min.

## Business Tie-in
- Use posterior probabilities for risk thresholding (e.g., approve if expected loss < threshold).
- Pairs well with policy simulators for operational value.