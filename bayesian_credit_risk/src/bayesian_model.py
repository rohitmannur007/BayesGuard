import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random

def hierarchical_logistic_model(X, groups, y=None, num_groups=None):
    # Priors
    mu_alpha = numpyro.sample('mu_alpha', dist.Normal(0, 10))  # Global intercept mean
    sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(5))  # Group intercept SD
    alpha_group = numpyro.sample('alpha_group', dist.Normal(mu_alpha, sigma_alpha), sample_shape=(num_groups,))
    
    beta = numpyro.sample('beta', dist.Normal(0, 5), sample_shape=(X.shape[1],))  # Coefficients
    
    # Group index (assume groups are integer-encoded 0 to num_groups-1)
    group_idx = jnp.array(groups)
    
    # Linear predictor
    logit_p = alpha_group[group_idx] + jnp.dot(X, beta)
    
    # Likelihood
    numpyro.sample('obs', dist.Bernoulli(logits=logit_p), obs=y)

def run_inference(model, X, groups, y, num_groups, rng_key=0, num_warmup=500, num_samples=1000, num_chains=4):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(random.PRNGKey(rng_key), X=X, groups=groups, y=y, num_groups=num_groups)
    return mcmc