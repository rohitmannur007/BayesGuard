import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
import numpyro
from numpyro.infer import Predictive
import jax.numpy as jnp

def posterior_predictive(mcmc, X, groups, num_groups, rng_key=0):
    predictive = Predictive(mcmc.sampler.model, mcmc.get_samples())
    post_pred = predictive(random.PRNGKey(rng_key), X=X, groups=groups, num_groups=num_groups)['obs']
    return post_pred

def evaluate_model(mcmc, X_test, groups_test, y_test, num_groups):
    # Get posterior samples
    samples = mcmc.get_samples()
    
    # Posterior predictive
    post_pred = posterior_predictive(mcmc, X_test, groups_test, num_groups)
    
    # Mean prob and uncertainty
    pred_prob = np.mean(post_pred, axis=0)
    pred_uncertainty = np.std(post_pred, axis=0)
    
    # Metrics
    auc = roc_auc_score(y_test, pred_prob)
    brier = brier_score_loss(y_test, pred_prob)
    
    print(f"AUC: {auc:.3f}, Brier Score: {brier:.3f}")
    
    return pred_prob, pred_uncertainty

def plot_calibration(y_test, pred_prob):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, pred_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title('Calibration Plot')
    plt.show()

def plot_ppc(post_pred, y_test):
    az.plot_ppc(az.from_numpyro(post_pred=post_pred, observed_data={'obs': y_test}))
    plt.show()

def expected_loss_threshold(pred_prob, pred_uncertainty, loss_threshold=0.1):
    # Example policy: Approve if expected loss (prob_default * uncertainty-adjusted) < threshold
    expected_loss = pred_prob * (1 + pred_uncertainty)  # Simple adjustment
    decisions = expected_loss < loss_threshold
    print(f"Approval rate: {np.mean(decisions):.2%}")
    return decisions