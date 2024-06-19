import itertools
import argparse
from functools import partial
from glob import glob
from typing import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd
import numpy as np

from Bio.Data.IUPACData import protein_letters, protein_letters_3to1
import torch
import esm
from esm.inverse_folding.util import CoordBatchConverter

Array = np.array

@dataclass
class PDBData:
  coords: Array
  sequence: str
  chain: str = 'A'
  @property
  def nres(self):
    return len(self.sequence)

def get_coords_and_seq(path, chain_id):
  structure = esm.inverse_folding.util.load_structure(path, chain_id)
  # set random residue names to UNK
  aa3s = [k.upper() for k in protein_letters_3to1.keys()]
  fix_aas = np.vectorize(lambda s: s if (s in aa3s) else 'UNK')
  structure.res_name = fix_aas(structure.res_name)
  coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
  return PDBData(coords, native_seq, chain_id)

def insert_padding_seq(sequences: Sequence[str], pad_length=10):
  return ('<mask>'*pad_length).join(sequences)

def insert_padding_coords(coords: Sequence[Array], pad_length=10):
  pad_coords = np.full((pad_length,3,3), float('inf'))
  return np.concatenate(list(itertools.chain(*[[v] + [pad_coords] for v in coords]))[:-1]) # skip the last bit

def get_logit_differences(logits, true_idx):
  true_logit = np.array([logit[idx] for logit, idx in zip(logits.T, true_idx)])
  logit_dif = (logits - true_logit)
  return logit_dif

def get_batch(path, chains, reverse_chains=False, pad_length=10, af2_structure=False, sequence_altering_fn=lambda x: x):
  if reverse_chains:
    chains = chains[::-1]
  pdb_data = [get_coords_and_seq(path, k) for k in chains]

  # opportunity to overrule what the pdb says
  for d in pdb_data:
    d.sequence = sequence_altering_fn(d.sequence)

  sequences_list = [a.sequence for a in pdb_data]
  coords_list = [a.coords for a in pdb_data]
  seq = insert_padding_seq(sequences_list, pad_length)
  coords = insert_padding_coords(coords_list, pad_length).astype('float32')
  entry = (coords, None, seq)
  return entry, pdb_data

def predict_single_residues(batch,  model, alphabet):
  batch_converter = CoordBatchConverter(alphabet)
  coords, confidence, strs, tokens, padding_mask = batch_converter(batch)
  prev_output_tokens = tokens[:, :-1]
  with torch.no_grad():
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
  token_probs = logits.detach().numpy()
  sequence_mask = (tokens[:,1:] != alphabet.mask_idx)
  true_idx = tokens[:,1:][sequence_mask].detach().numpy()
  return token_probs, true_idx

def get_logits(batch, model, alphabet, af2_structure=False):
  batch_converter = CoordBatchConverter(alphabet)
  coords, confidence, strs, tokens, padding_mask = batch_converter(batch)
  if af2_structure:
    # substitute cath token for af2
    tokens[0,0] = alphabet.get_idx('<af2>')

  prev_output_tokens = tokens[:, :-1]

  with torch.no_grad():
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

  token_probs = logits[0].detach().numpy()
  sequence_mask = (tokens[0,1:] != alphabet.mask_idx)
  if (not all(sequence_mask)):
    token_probs = token_probs[..., sequence_mask]
  true_idx = tokens[0,1:][sequence_mask].detach().numpy()
  return token_probs, true_idx

def predict_single(entry, model, alphabet):
  singles_batch = [(c[None], None, s) for c,s in zip(entry[0], entry[2])]
  return predict_single_residues(singles_batch, model, alphabet)[0].squeeze(-1).T

def get_predictions(path, chains, model, alphabet, reverse_chains=False, pad_length=10, af2_structure=False, sequence_altering_fn=lambda x: x):
  entry, pdb_data = get_batch(path, chains, reverse_chains=reverse_chains, pad_length=pad_length, af2_structure=af2_structure, sequence_altering_fn=sequence_altering_fn)
  logits, true_tokens = get_logits([entry], model, alphabet, af2_structure=af2_structure)
  logits_single_residue = predict_single(entry, model, alphabet)
  return logits, logits_single_residue, true_tokens, pdb_data

def predict_from_chain_mapping(path, chain_mapping, model, alphabet, af2_structure=False, pad_length=10, sequence_altering_fn=lambda x: x):
  predict = partial(get_predictions, model=model, alphabet=alphabet, af2_structure=af2_structure, pad_length=pad_length, sequence_altering_fn=sequence_altering_fn)
  '''
  as model is autoregressive, we predict heavy chain by running [...,VL,VH] so there is the context of the VL to predict VH
  and then [...,VH,VL], then combine.
  '''
  chains = list(chain_mapping.keys())
  logits, logits_single_residue, tokens, data = predict(path, chains)

  logit_differences = get_logit_differences(logits, tokens)
  logit_differences_single = get_logit_differences(logits_single_residue, tokens)
  df = pd.DataFrame({
    'post': alphabet.all_toks*logits.shape[-1],
    'logit_difference': logit_differences.T.ravel(),
    'logit_difference_single': logit_differences_single.T.ravel(),
    'pre': np.array(list(''.join([d.sequence for d in data]))).repeat(len(alphabet.all_toks)),
    'chain': np.array(list(itertools.chain(*[[d.chain] * len(d.sequence) for d in data]))).repeat(len(alphabet.all_toks)),
    'pos': np.array(list(itertools.chain(*[np.arange(len(d.sequence))+1 for d in data]))).repeat(len(alphabet.all_toks)),
  })
  df = df[df.post.apply(lambda s: s in protein_letters)]
  df['section'] = df.chain.apply(chain_mapping.__getitem__)
  return df

if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Reproduce predictions for ProteinMPNN on various datasets')
  parser.add_argument("--datasets_folder", help="Path to datasets folder", required=True)
  parser.add_argument("--datasets", nargs='+', help="Names of datasets to predict", default=['tsuboyama','s2648','s669'])
  parser.add_argument("--esmif_model_path", help="Path to ESMif model, `esm_if1_gvp4_t16_142M_UR50.pt`", required=True)
  args = parser.parse_args()

  f = args.datasets_folder
  datasets = args.datasets
  model_path = args.esmif_model_path

  model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
  model.eval()

  for dataset_name in datasets:
    pdb_folder = f'{f}/{dataset_name}/pdb/'
    experimental_data_path = f'{f}/{dataset_name}/{dataset_name}_ddg_experimental.csv'
    outpath = f'{f}/{dataset_name}/esmif_{dataset_name}.csv'

    d = pd.read_csv(experimental_data_path)
    d['pdb_path'] =pdb_folder+d.relative_path
    dfs = []
    for path, chain in d.groupby(['pdb_path', 'chain']).first().index:
      df = predict_from_chain_mapping(path, {chain:chain}, model, alphabet)
      df['pdb_path'] = path
      dfs.append(df)

    df = pd.concat(dfs)
    df = df[df.pre.apply(lambda s: s in 'ARNDCQEGHILKMFPSTWYV')]
    df['relative_path'] = df.pdb_path.apply(lambda s: s.split('/')[-1])

    # adhoc fix for numbering mismatchs
    if (dataset_name=='s2648'):
      m = df.pdb_path.apply(lambda s: any([k in s.lower() for k in ('1am7','1onc')]))
      df.loc[m, 'pos'] += 1
    elif (dataset_name=='s669'):
      d['pos'] = d.Mut_seq.apply(lambda s: int(s[1:-1]))
      m = df.relative_path.apply(lambda s: any([k in s.upper() for k in ('3DV0',)]))
      df.loc[m, 'pos'] += 1
    df = df[['pre','pos','post','relative_path','logit_difference', 'logit_difference_single']].rename(columns={'logit_difference':'ESMif'})
    df['ESMif-ddG-uncorrected'] = df['ESMif'] - df['logit_difference_single']
    d = d.merge(df, on=['pre','pos','post','relative_path'], how='left')

    if (dataset_name=='s669'):
      # switch back to inkeeping numbering
      d['pos'] = d.variant.apply(lambda s: int(s[1:-1]))

    d[['pdb_id','chain','pre','pos','post','ddG_experimental','ESMif','ESMif-ddG-uncorrected']].to_csv(outpath, index=False)
