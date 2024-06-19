import argparse
from functools import partial

import numpy as np
import pandas as pd

import jax
from jax import numpy as jnp, jit, grad, vmap
from evosax import CMA_ES
AA = list('ARNDCQEGHILKMFPSTWYV')

spearmanr_jax = lambda x, y: jnp.corrcoef(x.argsort().argsort(),y.argsort().argsort())[0,1]
pearsonr_jax = lambda x, y: jnp.corrcoef(x,y)[0,1]

def train_logits_cmaes(fitness_fn, seed=42, num_generations = 100, cmaes_kwargs=dict(popsize=50, num_dims=20, elite_ratio=0.5), shift_to_zero_mean=True):
  # Instantiate the search strategy
  rng = jax.random.PRNGKey(seed)
  strategy = CMA_ES(**cmaes_kwargs)
  es_params = strategy.default_params
  state = strategy.initialize(rng, es_params)

  # Run ask-eval-tell loop - NOTE: By default minimization!
  for t in range(num_generations):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state, es_params)
    fitness = fitness_fn(x)
    state = strategy.tell(x, fitness, state, es_params)

  # Get best overall population member & its fitness
  fitted_coeff = state.best_member if (not shift_to_zero_mean) else (state.best_member-state.best_member.mean())
  return (fitted_coeff, state.best_fitness)

i = AA.index('M')
def apply_M_correction(M_coeff, coeff_esmif):
  return jnp.array(coeff_esmif).at[:,i].add(-M_coeff).at[i,:].add(M_coeff)

def fitness_fn(M_coeff, coeff_esmif, coeff_proteinmpnn):
  v = apply_M_correction(M_coeff, coeff_esmif)
  return -jnp.corrcoef(v.ravel(), coeff_proteinmpnn.ravel())[0,1]

if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Fit coefficient to correct for methionine bias in first residue predictions of ESMif')
  parser.add_argument("--data_folder", help="Folder containing `coeff_esmif_raw.csv` and `coeff_proteinmpnn_ddg_v_48_020.csv`, csv files containing (20,20) matrix of mean delta X->Y for each of the 20 amino acids ", default='data/')
  args = parser.parse_args()

  f = args.data_folder
  esmif_path = f'{f}/coeff_esmif_raw.csv'
  proteinmpnn_path = f'{f}/coeff_proteinmpnn_ddg_v_48_020.csv'
  corrected_esmif_path = f'{f}/coeff_esmif.csv'

  d_proteinmpnn, d_esmif = [pd.read_csv(path, index_col=0)[AA].loc[AA] for path in (proteinmpnn_path, esmif_path)]
  closed_fitness_fn = partial(fitness_fn, coeff_esmif=d_esmif.values, coeff_proteinmpnn=d_proteinmpnn.values)
  fitted_coeff, loss = train_logits_cmaes(jax.jit(jax.vmap(closed_fitness_fn)), cmaes_kwargs=dict(popsize=50, num_dims=1, elite_ratio=0.5), shift_to_zero_mean=False)
  print(f"Fitted coefficient to correct for methionine bias in first residue predictions of ESMif: {fitted_coeff[0]:.2f}")
  coeff_esmif_fitted = apply_M_correction(fitted_coeff, d_esmif.values)
  d_esmif_fitted = pd.DataFrame(coeff_esmif_fitted, index=AA,
          columns=AA)

  d = pd.concat([d.stack().rename(name) for d,name in [(d_esmif,'esmif'),(d_proteinmpnn, 'proteinmpnn'), (d_esmif_fitted, 'esmif_methionine_corrected')]], axis=1)
  d = d[[(x[0]!=x[1]) for x in d.index]]
  d_not_M = d[[('M' not in x) for x in d.index]]
  corrs_not_M = d_not_M.corr('pearson')
  print(f"ProteinMPNN-ESMif ddG coefficients (mutations involving methionine removed), pearson correlation: {corrs_not_M.loc['esmif','proteinmpnn']:.3f}")
  corrs = d.corr('pearson')
  print(f"ProteinMPNN-ESMif ddG coefficients pre-methionine correction, pearson correlation: {corrs.loc['esmif','proteinmpnn']:.3f}")
  print(f"ProteinMPNN-ESMif ddG coefficients post-methionine correction, pearson correlation: {corrs.loc['esmif_methionine_corrected','proteinmpnn']:.3f}")
  d_esmif_fitted.to_csv(corrected_esmif_path)
