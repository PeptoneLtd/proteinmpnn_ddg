import argparse
import csv
from dateutil import parser as time_parser
from os.path import exists
from functools import partial

import pandas as pd
import numpy as np

import torch
import jax
from jax import jit, vmap, numpy as jnp

from proteinmpnn_ddg import mk_mpnn_model, pad, ALPHABET, ALPHABET_CLASSIC, RunModel, _aa_convert, get_compute_logit_differences_fn

def build_training_clusters(params, debug):
  # Taken from ProteinMPNN repo, training/utils.py, https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/training/utils.py

  val_ids = set([int(l) for l in open(params['VAL']).readlines()])
  test_ids = set([int(l) for l in open(params['TEST']).readlines()])

  if debug:
    val_ids = []
    test_ids = []

  # read & clean list.csv
  with open(params['LIST'], 'r') as f:
    reader = csv.reader(f)
    next(reader)
    rows = [[r[0],r[3],int(r[4])] for r in reader
        if float(r[2])<=params['RESCUT'] and
        time_parser.parse(r[1])<=time_parser.parse(params['DATCUT'])]

  # compile training and validation sets
  train = {}
  valid = {}
  test = {}

  if debug:
    rows = rows[:20]
  for r in rows:
    if r[2] in val_ids:
      if r[2] in valid.keys():
        valid[r[2]].append(r[:2])
      else:
        valid[r[2]] = [r[:2]]
    elif r[2] in test_ids:
      if r[2] in test.keys():
        test[r[2]].append(r[:2])
      else:
        test[r[2]] = [r[:2]]
    else:
      if r[2] in train.keys():
        train[r[2]].append(r[:2])
      else:
        train[r[2]] = [r[:2]]
  if debug:
    valid=train
  return train, valid, test

get_n_chunks = lambda m, chunk_size: m//chunk_size+ (1 if ((m%chunk_size)!=0) else 0)
def run_batched(inputs, run_fn, chunk_size, verbose=True):
  lengths = jax.tree.flatten(jax.tree.map(lambda x: jnp.shape(x)[0], inputs))[0]
  assert len(set(lengths))==1
  n_batches = get_n_chunks(lengths[0], chunk_size)

  # infer outputs and setup empty numpy arrays to transfer to
  # numpy arrays are flat list, then mapped back to output shapes at end
  args = jax.tree.map(lambda x: x[:1], inputs)
  o = run_fn(*args)
  n = lengths[0]
  o_flat, tree_defn = jax.tree.flatten(o)
  outputs = [np.empty((n,)+jnp.shape(x)[1:], dtype=x.dtype) for x in o_flat]
  for i in range(n_batches):
    args = jax.tree.map(lambda x: x[i*chunk_size:(i+1)*chunk_size], inputs)
    o = run_fn(*args)
    for j, vals in enumerate(jax.tree.flatten(o)[0]):
      outputs[j][i*chunk_size:(i+1)*chunk_size] = vals
    if (verbose):
      print(f'{(i+1)/n_batches:.0%} complete')
  outputs = jax.tree.unflatten(tree_defn, outputs)
  return outputs

aa1_to_idx = np.vectorize({v:k for k,v in dict(enumerate(ALPHABET)).items()}.__getitem__)

def build_input(pdb_path):
  entry = torch.load(pdb_path)
  mask = entry['mask'][...,:4].all(-1).numpy()
  X = entry['xyz'][:,:4,:].numpy()

  s = np.array(entry['seq'])[None].view('U1')
  # replace unknown chars with X
  s = np.vectorize(lambda s: s if (s in ALPHABET) else 'X')(s)
  S = aa1_to_idx(s)
  n = X.shape[0]
  I = {
    'X': X,
    'S': S,
    'residue_idx': np.arange(n, dtype=int)+1,
    'mask': mask.astype(float),
    'chain_idx': np.zeros(n, dtype=int),
  }
  return I

def process_to_single_residue(I):
  mask = (I['mask']==1.)
  I = {k:v[mask][:,None] for k,v in I.items()}
  return I

def build_proteinmpnn_splits(data_path):
  params = {
    "LIST"  : f"{data_path}/list.csv",
    "VAL"   : f"{data_path}/valid_clusters.txt",
    "TEST"  : f"{data_path}/test_clusters.txt",
    "DIR"   : f"{data_path}",
    "DATCUT"  : "2030-Jan-01",
    "RESCUT"  : 3.5, #resolution cutoff for PDBs
    "HOMO"  : 0.70 #min seq.id. to detect homo chains
  }
  train, valid, test = build_training_clusters(params, False)
  d = pd.read_csv(params['LIST'])
  cluster_d = pd.concat([
      pd.DataFrame({'CLUSTER':v, 'data_split': k})
      for k,v in {'train':train.keys(),
            'test':test.keys(),
            'valid':valid.keys()
          }.items()])
  d = d.merge(cluster_d)
  return d

def compute_ddg_corrections(d, model_name, batch_size=100000):
  key = jax.random.key(42)
  model = mk_mpnn_model(model_name)
  compute_logit_differences_fn = get_compute_logit_differences_fn(model)
  compute_logit_differences_fn_vmap = jit(vmap(partial(compute_logit_differences_fn, key=key)))

  Is = list(d[d.data_split=='train'].pdb_path.apply(build_input))
  Is_single = list(map(process_to_single_residue, Is))
  Is_single_flat = jax.tree.map(lambda *args: np.concatenate(args, axis=0), *Is_single)
  logit_differences = run_batched((Is_single_flat,), compute_logit_differences_fn_vmap, batch_size, verbose=False).squeeze(1)

  S = Is_single_flat['S'].squeeze(1)
  naa = logit_differences.shape[-1]
  mean_logit_difference = jnp.zeros((naa, naa)).at[S].add(logit_differences) / jnp.zeros((naa, naa)).at[S].add(1.)
  ddg_correction_df = pd.DataFrame(
    np.round(mean_logit_difference, 3),
    index=list(ALPHABET),
    columns=list(ALPHABET)
  ).loc[list(ALPHABET_CLASSIC)][list(ALPHABET_CLASSIC)]
  return ddg_correction_df, Is_single_flat

def apply_logits_fn(I, compute_logits_fn, key):
  I['X'] = np.nan_to_num(I['X'])
  n = I['S'].shape[0]
  if (I['mask'].sum()>48):
    I = {k:pad(v, fill_value=0) for k,v in I.items() if k in ['X','mask','residue_idx','chain_idx','S',]}
  return (I['S'][:n],
     compute_logits_fn(I, key)[...,:n,:])

def compute_effect_of_decode_last(d, model_name, seed=42):
  key = jax.random.PRNGKey(seed)

  Is = list(d[d.data_split=='valid'].pdb_path.apply(build_input))
  model = mk_mpnn_model(model_name)
  compute_logits_random_order, compute_logits_decode_last = [get_compute_logit_differences_fn(model, work_efficient) for work_efficient in [False,True]]
  def predict_logits_both_ways(I, key):
    S, logits_random = apply_logits_fn(I, compute_logits_random_order, key)
    _, logits_decode_last = apply_logits_fn(I, compute_logits_decode_last, key)
    mask = I['mask'].astype(bool)
    return {k: v[mask] for k,v in {'S':S, 'logits_random': logits_random, 'logits_decode_last': logits_decode_last}.items()}

  keys = jax.random.split(key, len(Is))

  outputs = list(map(lambda args: predict_logits_both_ways(*args), zip(Is, keys)))
  outputs_concat = jax.tree.map(lambda *args: np.concatenate(args, axis=0), *outputs)
  mask = outputs_concat['S']<20
  outputs_concat = {k: v[mask] for k,v in outputs_concat.items()}

  seq_recoveries = {k:(outputs_concat[k][:,:20].max(1)==0.).mean() for k in ['logits_random', 'logits_decode_last',]}
  def compute_ece(o, logit_key):
    ce = jax.nn.log_softmax(o[logit_key][:,:20])
    return jnp.exp(-vmap(lambda x,y: x[y])(ce, o['S']).mean())
  exponentiated_cross_entropies = {k: compute_ece(outputs_concat,k) for k in ['logits_random', 'logits_decode_last',]}
  for prefix, k in [('Classic ProteinMPNN with single random order', 'logits_random'), ('ProteinMPNN with each residue decoded last', 'logits_decode_last')]:
    print(f'{prefix}, Sequence recovery: {seq_recoveries[k]:.1%}, Exponentiated Cross Entropy: {exponentiated_cross_entropies[k]:.2f}')
  return d[d.data_split=='valid'], Is, outputs

if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Reproduce predictions for ProteinMPNN on various datasets')
  parser.add_argument("--data_path", help="Folder of pdb structures saved as .pt files, taken from `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz` and extracted", required=True)
  parser.add_argument("--structure_data_outpath", help="Path to write a processed set of pdb structures from .pt to .npz file format for computing the ESMif-ddG coefficients with", default='training_single_structure_per_cluster_23349_structures_5615050_residues')
  parser.add_argument("--outpath", default=None, help='Path to write ProteinMPNN-ddG coefficients to', required=True)
  args = parser.parse_args()

  data_path = args.data_path
  structure_data_outpath = args.structure_data_outpath
  outpath = args.outpath
  model_name = 'v_48_020'
  # data_path = 'pdb_2021aug02/'
  # outpath = 'coeff_proteinmpnn_ddg_v_48_020.csv'
  # structure_data_outpath = 'training_single_structure_per_cluster_23349_structures_5615050_residues'
  batch_size=100000

  d_all = build_proteinmpnn_splits(data_path)
  d = d_all.groupby('CLUSTER').sample(1, random_state=42)
  d['pdb_path'] = d.CHAINID.apply(lambda s: f'{data_path}/pdb/{s[1:3]}/{s}.pt')
  # assert d.pdb_path.apply(exists).all()

  '''A single structure was taken for each of the
  23,349 cluster in the training set of ProteinMPNN, residues
  for which the backbone atom positions are unknown were
  removed resulting in 5,615,050 single residue background
  geometries with at least 77,000 geometries for each distinct
  amino acid.'''
  ddg_correction_df, Is_single_flat = compute_ddg_corrections(d, model_name, batch_size=batch_size)
  ddg_correction_df.to_csv(outpath)

  # process files for ESMif to predict based on single residue
  Is_single_flat['s'] = np.array(ALPHABET)[None].view('U1')[Is_single_flat['S']]
  np.savez(structure_data_outpath,
      **Is_single_flat)

  ''' Improved sequence recovery metrics from modifications of ProteinMPNN. A single structure for each of the 1,464
  clusters in the ProteinMPNN validation set were taken and predicted.
  *Exponentiated cross-entropy computed over 20 natural amino acids, averaged per position'''
  compute_effect_of_decode_last(d, model_name, seed=42)

  '''
  We found the log-odds ratios calculated from the amino acid frequencies in the training set of ProteinMPNN correlate well with those observed
  '''
  aa_background_probs = jnp.zeros(21).at[Is_single_flat['S'].ravel()].add(1)[:20]
  aa_frequency_logit_correction = pd.Series(dict(zip(ALPHABET_CLASSIC, np.log(aa_background_probs/aa_background_probs.mean())))).sort_values().apply(lambda x: np.round(x,3)).to_dict()
  df = pd.read_csv(outpath, index_col=0)
  df = pd.DataFrame(df.stack().rename('proteinmpnn_correction'))
  df['background_aa_frequencies'] = [aa_frequency_logit_correction[post]-aa_frequency_logit_correction[pre] for pre, post in df.index]
  df = df[[(x[0]!=x[1]) for x in df.index]] # remove null mutations where pre==post
  corr = df.corr('spearman').loc['proteinmpnn_correction','background_aa_frequencies']
  print(f"Spearman correlation between ProteinMPNN single residue context predictions and the background amino acid frequencies: {corr:.2f}")
