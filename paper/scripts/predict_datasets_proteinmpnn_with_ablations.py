import argparse
from glob import glob

import numpy as np
import pandas as pd

import jax
from jax import jit, vmap, numpy as jnp
from proteinmpnn_ddg import mk_mpnn_model, RunModel, _aa_convert, load_model_and_predict_functions, ALPHABET, ALPHABET_CLASSIC, pad

def read_pdb(pdb_path):
  d = pd.read_fwf(pdb_path, widths=[6,5,1,4,1,3,1,1,4,2,2,8,8,8,2,4,1,5,2,2,2,4,2,1], header=None, delimiter=None)
  d = d[d[0]=='ATOM'][[0,1,3,5,7,8,11,12,13,15,17,22]]
  d.columns=['recordtype','atomnumber','atomname','resname','chainname','resnumber','x','y','z','occupancy','bfactor','element']
  d = d.astype({'atomnumber':int, 'resnumber':int})
  return d

def get_compute_logit_differences_fn_single_decoding_order(model):
  run_model = RunModel(model._model.config)
  run_model.params = model._model.params
  @jit
  def _compute_logit_differences(I, key):
  nres = I['S'].shape[0]
  I['S'] = _aa_convert(I['S']).argmax(-1)
  I['decoding_order'] = jax.random.uniform(key, (nres,)).argsort()
  o = run_model.score(run_model.params, key, I)
  logit_differences = jax.vmap(lambda x,y: x-x[y])(o['logits'],o['S'])
  return _aa_convert(logit_differences, rev=True)
  return _compute_logit_differences

model, compute_logit_differences_fn_vmap, compute_logit_differences_fn_single_residue = load_model_and_predict_functions('v_48_020')
compute_logit_differences_fn_single_decoding_order = get_compute_logit_differences_fn_single_decoding_order(model)

def compute_logit_differences_for_pdb_path(pdb_path, key, context_chains=None, pad_inputs=True, nrepeat=1):
  if (context_chains is None):
    context_chains = list(read_pdb(pdb_path).chainname.unique())
    assert len(context_chains)==1
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
  logit_differences_single_residue = compute_logit_differences_fn_single_residue(I_padded)[:n]
  logit_differences_single_decoding_order = compute_logit_differences_fn_single_decoding_order(I_padded, key)[:n]

  df = pd.DataFrame({
      'pre':pre_amino_acids, 'post':post_amino_acids, 'pos':positions,
      'ProteinMPNN+decode last':logit_differences.mean(0)[chain_mask].ravel(),
      'ProteinMPNN': logit_differences_single_decoding_order[chain_mask].ravel()
    })
  df['ProteinMPNN-ddG'] = df['ProteinMPNN+decode last'] - logit_differences_single_residue[chain_mask].ravel()
  df = df[df.post.apply(lambda s: s in ALPHABET_CLASSIC)]
  return df

if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Reproduce predictions for ProteinMPNN on various datasets')
  parser.add_argument("--datasets_folder", help="Path to datasets folder", required=True)
  parser.add_argument("--datasets", nargs='+', help="Names of datasets to predict", default=['tsuboyama','s2648','s669'])
  args = parser.parse_args()

  f = args.datasets_folder
  datasets = args.datasets

  for dataset_name in datasets:
    pdb_folder = f'{f}/{dataset_name}/pdb/'
    experimental_data_path = f'{f}/{dataset_name}/{dataset_name}_ddg_experimental.csv'
    outpath = f'{f}/{dataset_name}/proteinmpnn_{dataset_name}.csv'

    key = jax.random.PRNGKey(42)

    d = pd.read_csv(experimental_data_path)
    d['pdb_path'] =pdb_folder+d.relative_path

    dfs = []
    entries = d.groupby(['pdb_path', 'chain']).first().index
    keys = jax.random.split(key, len(entries))
    for (path, chain), key in zip(entries, keys):
      print(path, chain)
      df = compute_logit_differences_for_pdb_path(path, key, context_chains=[chain])
      df['pdb_path'] = path
      dfs.append(df)

    df = pd.concat(dfs)
    df = df[df.pre.apply(lambda s: s in 'ARNDCQEGHILKMFPSTWYV')]
    df['relative_path'] = df.pdb_path.apply(lambda s: s.split('/')[-1])
    d = d[['pdb_id','chain','pre','pos','post','ddG_experimental','relative_path']].merge(df.drop(columns=['pdb_path']), on=['pre','pos','post','relative_path'], how='left')
    d[['pdb_id', 'chain', 'pre', 'pos', 'post', 'ddG_experimental',
      'ProteinMPNN+decode last', 'ProteinMPNN',
      'ProteinMPNN-ddG']].to_csv(outpath, index=False)
