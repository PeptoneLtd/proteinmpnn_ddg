from typing import List

import numpy as np
import pandas as pd

import jax
from jax import vmap, jit, numpy as jnp

from colabdesign.mpnn import mk_mpnn_model
from proteinmpnn_ddg.work_efficient_decode_last import get_compute_logit_differences_fn

ALPHABET = 'ARNDCQEGHILKMFPSTWYVX'
ALPHABET_CLASSIC = ALPHABET[:-1]

def load_model_and_predict_functions(model_name):
  model = mk_mpnn_model(model_name)
  compute_logit_differences_fn = get_compute_logit_differences_fn(model)
  compute_logit_differences_fn_vmap = jit(vmap(compute_logit_differences_fn, in_axes=(None,0)))
  compute_logit_differences_fn_single_residue = lambda I: jit(vmap(compute_logit_differences_fn_vmap, in_axes=(0,None)))(
    jax.tree.map(lambda x: x[:,None], {k:v for k,v in I.items() if k in ['X','mask', 'S', 'residue_idx', 'chain_idx']}),
    jax.random.PRNGKey(42)[None]
  ).squeeze((1,2)) # pregiven key as there is no randomness in these predictions
  return (model, compute_logit_differences_fn_vmap, compute_logit_differences_fn_single_residue)
MODEL_CACHE = {k: load_model_and_predict_functions(k) for k in ['v_48_020']}

def pad(x, n=None, fill_value=0):
  m = x.shape[0]
  if (n is None):
    n = 2**int(np.ceil(np.log2(x.shape[0])))
  return np.concatenate([x, np.full((n-m,)+x.shape[1:], fill_value=fill_value, dtype=x.dtype)], dtype=x.dtype)

def predict_logits_for_all_point_mutations_of_single_pdb(
    model_name:str,
    context_chains:List[str],
    pdb_path:str,
    nrepeat: int=5,
    seed: int=42,
    chain_to_predict:str=None,
    pad_inputs=False,
    apply_ddG_correction=True
    ):
  key = jax.random.PRNGKey(seed)

  if (chain_to_predict is None):
    chain_to_predict = context_chains[0] # assume first chain
  if (chain_to_predict in context_chains):
    context_chains.remove(chain_to_predict)
  context_chains.insert(0, chain_to_predict)

  if (model_name not in MODEL_CACHE):
    MODEL_CACHE[model_name] = load_model_and_predict_functions(model_name)
  model, compute_logit_differences_fn_vmap, compute_logit_differences_fn_single_residue = MODEL_CACHE[model_name]
  model.prep_inputs(pdb_filename=pdb_path, chain=','.join(context_chains))
  I = model._inputs

  chain_mask = (I['chain_idx']==0)
  positions = I['residue_idx'][chain_mask]
  seq = np.vectorize(ALPHABET.__getitem__)(I['S'][chain_mask])
  nres = chain_mask.sum()
  pre_amino_acids = np.repeat(seq, len(ALPHABET))
  post_amino_acids = np.tile(list(ALPHABET), nres)
  positions = np.repeat(positions, len(ALPHABET))

  n = I['S'].shape[-1]
  if (pad_inputs and (n>48)):
    I_padded = {k:pad(v, fill_value=0) for k,v in I.items() if k in ['X','mask','residue_idx','chain_idx','S',]}
  else:
    if (pad_inputs and not (n>48)):
      print("Padding ineffective when nresidue<49, due to non-identity ops for padded residues, Running unpadded")
    I_padded = I
  logit_differences = compute_logit_differences_fn_vmap(I_padded, jax.random.split(key, nrepeat))[...,:n,:]
  df = pd.DataFrame({
      'pre':pre_amino_acids, 'post':post_amino_acids, 'pos':positions,
      'logit_difference':logit_differences.mean(0)[chain_mask].ravel(),
    })
  if (apply_ddG_correction):
    df[f'logit_difference_ddg'] = df['logit_difference'].values - compute_logit_differences_fn_single_residue(I_padded)[:n][chain_mask].ravel()

  df = df[df.post.apply(lambda s: s in ALPHABET_CLASSIC)]
  return df
