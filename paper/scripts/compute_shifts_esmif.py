import argparse

import numpy as np
import pandas as pd

import torch
import esm
from esm.inverse_folding.util import CoordBatchConverter
get_n_chunks = lambda m, chunk_size: m//chunk_size+ (1 if ((m%chunk_size)!=0) else 0)
protein_letters = 'ACDEFGHIKLMNPQRSTVWY'

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

if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Reproduce predictions for ProteinMPNN on various datasets')
  parser.add_argument("--structure_data_path", help="Path of the .npz file of pre-filtered structure dataset which `compute_shifts.py` script generates", required=True)
  parser.add_argument("--esmif_model_path", help="Path to ESMif model, `esm_if1_gvp4_t16_142M_UR50.pt`", required=True)
  parser.add_argument("--outpath", default=None, help='Path to write ESMif-ddG coefficients to', required=True)
  args = parser.parse_args()

  structure_data_path = args.structure_data_path
  model_path = args.esmif_model_path
  outpath = args.outpath

  # model_path='esm_if1_gvp4_t16_142M_UR50.pt'
  # structure_data_path = 'training_single_structure_per_cluster_23349_structures_5615050_residues.npz'
  chunk_size=1000
  # outpath = 'coeff_esmif_raw.csv'

  model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
  model.eval()

  v = np.load(structure_data_path)
  batches = [(coords, None,seq) for coords, seq in zip(v['X'],v['s'].ravel())]
  n_batches = get_n_chunks(len(batches), chunk_size)

  outputs = []
  for i in range(n_batches):
    o = predict_single_residues(batches[i*chunk_size:(i+1)*chunk_size], model, alphabet)
    outputs.append(o)
    print(f'{(i+1)*chunk_size} completed')

  aa_indexs = np.array([alphabet.all_toks.index(k) for k in protein_letters])
  logits, S = [np.concatenate([o[i] for o in outputs]) for i in [0,1]]
  mask = (S<24) & (S>=4)
  # convert logits and S to only natural aas with their indexing in protein_letters
  logits = logits[mask][:, aa_indexs].squeeze(-1)
  S = S[mask]
  S = (S[:,None]==aa_indexs[None]).argmax(-1)

  true_logit = np.array([logit[idx] for logit, idx in zip(logits, S)])
  logit_dif = (logits - true_logit[...,None])

  # collate to averages
  x = np.zeros((20,20))
  counts = np.zeros((20,20))
  np.add.at(x, S, logit_dif)
  np.add.at(counts, S, 1)
  mean_logit_difference = x/counts

  ddg_correction_df = pd.DataFrame(
    np.round(mean_logit_difference, 3),
    index=list(protein_letters),
    columns=list(protein_letters)
  )
  ddg_correction_df.to_csv(outpath)
