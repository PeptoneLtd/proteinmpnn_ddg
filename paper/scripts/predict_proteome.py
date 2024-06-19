import argparse
import time
from glob import glob
from os.path import basename

import pandas as pd
import numpy as np

from jax import jit, vmap
import jax

from proteinmpnn_ddg import pad, ALPHABET, ALPHABET_CLASSIC, load_model_and_predict_functions
# or can use previous version to match RaSP - https://ftp.ebi.ac.uk/pub/databases/alphafold/v2/UP000005640_9606_HUMAN_v2.tar
s = '''
apt-get update && apt-get install -y aria2
aria2c -x 16 https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar
tar -xf UP000005640_9606_HUMAN_v4.tar --wildcards --no-anchored '*.pdb.gz'
gunzip *.pdb.gz
'''
idx_to_aa1 = np.vectorize(dict(enumerate(ALPHABET)).__getitem__)
aa3_to_idx = np.vectorize({'ALA': 0,
 'CYS': 4,
 'ASP': 3,
 'GLU': 6,
 'PHE': 13,
 'GLY': 7,
 'HIS': 8,
 'ILE': 9,
 'LYS': 11,
 'LEU': 10,
 'MET': 12,
 'ASN': 2,
 'PRO': 14,
 'GLN': 5,
 'ARG': 1,
 'SER': 15,
 'THR': 16,
 'VAL': 19,
 'TRP': 17,
 'TYR': 18}.__getitem__)
char_newline = np.array('\n').view('i4')
char_ATOM = np.array(['ATOM']).view('i4')
atomname_char4 = np.array([' N  ',' CA ',' C  ',' O  '])

pdb_fwf_cols = {'recordtype': (0, 6),
 'atomnumber': (6, 11),
 'atomname': (12, 16),
 'resname': (17, 20),
 'chainname': (21, 22),
 'resnumber': (22, 26),
 'resnumber_tag': (26, 28),
 'x': (30, 38),
 'y': (38, 46),
 'z': (46, 54),
 'occupancy': (56, 60),
 'bfactor': (61, 66),
 'element': (76, 78)}

def load_fast(pdb_path):
  x = np.fromfile(pdb_path, dtype=np.uint8)
  x.resize(x.shape[0]+1)
  l = x.reshape(-1,81)
  atom_mask = (l[:,:4]==char_ATOM).all(-1)
  # assume (N, then CA, then C, then O) in order
  NCACO_mask = (l[:,12:16].astype(np.int32).view('U4')==atomname_char4).any(1)
  l = l[atom_mask & NCACO_mask]
  X = l[:,30:54].astype(np.int32).view('U8').astype(np.float32).reshape(-1,4,3)
  S = aa3_to_idx(l[::4,17:20].astype(np.int32).view("U3")).ravel()
  n = S.shape[0]
  I = {
    'X': X,
    'S': S,
    'residue_idx': np.arange(n, dtype=int)+1,
    'mask': np.ones(n, dtype=float),
    'chain_idx': np.zeros(n, dtype=int),
  }
  return I

def predict_pdb(pdb_path, compute_logit_differences_ddg_fn, key):
  I = load_fast(pdb_path)
  n = I['S'].shape[0]
  if (n>48):
    I = {k:pad(v, fill_value=0) for k,v in I.items() if k in ['X','mask','residue_idx','chain_idx','S',]}
  return (I['S'][:n],
     compute_logit_differences_ddg_fn(I, key)[...,:n,:])

# shapes_to_compile = np.concatenate([
#   np.arange(16,49), 2**np.arange(6,13)])
# def precompile(compute_logit_differences_fn, shapes_to_compile=shapes_to_compile):
#   key = jax.random.PRNGKey(0)
#   n = shapes_to_compile.max()
#   I = {
#     'X': jax.random.uniform(key, (n,4,3))*10,
#     'S': np.zeros(n, dtype=int),
#     'residue_idx': np.arange(n, dtype=int)+1,
#     'mask': np.ones(n, dtype=float),
#     'chain_idx': np.zeros(n, dtype=int),
#   }
#   start = time.time()
#   results = {}
#   for i in np.sort(shapes_to_compile)[::-1]:
#     I = jax.tree.map(lambda x: x[:i], I)
#     results[i] = compute_logit_differences_fn(I, key)
#   _ = jax.block_until_ready(results)
#   end = time.time()
#   print(f'Total time: {end-start:.0f} seconds, to compile {shapes_to_compile.shape[0]} shapes')
def run_proteome(folder, seed):
  paths = glob(f'{folder}/*.pdb')
  key = jax.random.split(jax.random.PRNGKey(seed), 1)[0]
  model, compute_logit_differences_fn_vmap, compute_logit_differences_fn_single_residue = load_model_and_predict_functions('v_48_020')
  compute_logit_differences_ddg_fn = jit(lambda I, key: compute_logit_differences_fn_vmap(I, key[None]).squeeze(0) - compute_logit_differences_fn_single_residue(I))

  start = time.time()
  results = {}
  for pdb_path in paths:
    results[basename(pdb_path)[:-4]] = predict_pdb(pdb_path, compute_logit_differences_ddg_fn, key)
  _ = jax.block_until_ready(results)
  end = time.time()

  nres = np.array([v[0].shape[0] for v in results.values()])
  print(f'Total time: {end-start:.0f} seconds, approx {(end-start)/nres.sum()*1e6:.0f}us per position')

  # write out files
  for name, (S, logit_differences_ddg) in results.items():
    seq = idx_to_aa1(S)
    nres = S.shape[0]
    pre_amino_acids = np.repeat(seq, len(ALPHABET_CLASSIC))
    post_amino_acids = np.tile(list(ALPHABET_CLASSIC), nres)
    positions = np.repeat(np.arange(nres)+1, len(ALPHABET_CLASSIC))
    df = pd.DataFrame({
      'pre':pre_amino_acids, 'post':post_amino_acids, 'pos':positions,
      'logit_difference_ddg':logit_differences_ddg[...,:20].ravel(),
    })
    df.to_csv(f'{folder}/{name}.csv', index=False)


if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Uses ProteinMPNN-ddG to compute for all point mutations of all AlphaFold2 PDBs in a folder')
  parser.add_argument("--folder", help="Path to folder of pdbs", required=True)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  run_proteome(args.folder, args.seed)

  # compute_logit_differences_fn = get_compute_logit_differences_fn(mk_mpnn_model('v_48_020'))
  # precompile(compute_logit_differences_fn, shapes_to_compile=shapes_to_compile)
