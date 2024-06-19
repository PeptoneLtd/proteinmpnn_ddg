import argparse
import re
from functools import reduce, partial

import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

def compute_tp_fp_fn_tn(y_true, y_prob):
  n = y_prob.shape[0]
  pad_val = y_prob.max()
  _, counts = np.unique(y_prob, return_counts=True)
  order = np.argsort(y_prob)

  tp = (y_true.sum()-y_true[order].cumsum())[counts.cumsum()-1]
  tp = np.concatenate([np.array([y_true.sum()]), tp])

  np_ = n-np.concatenate([np.array([0]), counts.cumsum()])
  fp = np_ - tp
  fn = y_true.sum()-tp
  tn = n-y_true.sum()-fp
  return tp, fp, fn, tn

def compute_roc_auc(y_true, y_prob):
  tp, fp, fn, tn = compute_tp_fp_fn_tn(y_true, y_prob)
  fpr, tpr = (fp / (fp + tn), tp / (tp + fn))
  return -np.trapz(tpr, fpr)

def compute_pr_auc(y_true, y_prob):
  tp, fp, fn, tn = compute_tp_fp_fn_tn(y_true, y_prob)
  precision, recall = tp/(tp+fp), tp/(tp+fn)
  precision = np.concatenate([np.array([0]), precision])
  recall = np.concatenate([np.array([1]), recall])
  precision = np.nan_to_num(precision, nan=1.)
  return -(np.diff(recall)*precision[:-1]).sum()

def compute_metrics(d, cols):
  return pd.concat([
    pd.DataFrame(
      {k:
        {c:
           fn(d.ddG_experimental.values<0, d[c].values)
         for c in cols}
       for k, fn in [('roc_auc', compute_roc_auc), ('pr_auc', compute_pr_auc)]}
    ),
    -d[cols+['ddG_experimental']].corr('spearman')[cols].loc['ddG_experimental'].rename('spearman'),
    pd.Series(
      (d['ddG_experimental'].values<0)[d[cols].values.argsort(0)][-10:].mean(0),
      index=cols, name='top 10')
  ], axis=1)

def compute_metrics_with_groupby(d, cols, col='pdb_id'):
  return d.groupby(col).apply(partial(compute_metrics, cols=cols)).reset_index(level=-1).groupby('level_1').mean()

def read_and_join_csvs(paths):
  dfs = list(map(pd.read_csv, paths))
  return reduce(lambda x,y: x.merge(y, how='outer'), dfs)

def correct_esmif(d):
  # The 4.18 coeffient fix for ESMif, derived from correlating to ProteinMPNN
  d['ESMif-ddG'] = d['ESMif-ddG-uncorrected'] + 4.18*((d.post=='M').astype(int) - (d.pre=='M').astype(int))
  return d


if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Reproduce predictions for ProteinMPNN on various datasets')
  parser.add_argument("--datasets_folder", help="Path to datasets folder", default='datasets/')
  args = parser.parse_args()

  datasets_folder = args.datasets_folder

  results = {}

  ### Tsuboyama
  f = f'{datasets_folder}/tsuboyama/'
  paths = [
    f'{f}/proteinmpnn_tsuboyama.csv',
    f'{f}/esmif_tsuboyama.csv',
    f'{f}/other_predictors_tsuboyama.csv',
  ]
  dfs = list(map(pd.read_csv, paths))
  d = reduce(lambda x,y: x.merge(y), dfs)
  d = correct_esmif(d)
  d['RaSP'] *= -1 # inkeeping with other predictors
  cols = ['ProteinMPNN', 'ProteinMPNN+decode last',
    'ProteinMPNN-ddG', 'ESMif', 'ESMif-ddG',  'ACDC-NN-Seq', 'ACDC-NN-Struct','DDGun','DDGun3D','RaSP']
  results['tsuboyama'] = {
    'all': compute_metrics_with_groupby(d, cols),
    'no_methionine': compute_metrics_with_groupby(d[~((d.pre=='M') | (d.post=='M'))], cols)
  }

  ### 669
  f = f'{datasets_folder}/s669/'
  paths = [
    f'{f}/proteinmpnn_s669.csv',
    f'{f}/esmif_s669.csv',
    f'{f}/rasp_s669.csv',
    f'{f}/s669_ddg_experimental.csv',
  ]
  d = read_and_join_csvs(paths)
  d = correct_esmif(d)
  d = d[['pdb_id', 'chain', 'pre', 'pos', 'post', 'ddG_experimental',
    'ESMif','ESMif-ddG',
    'ProteinMPNN', 'ProteinMPNN-ddG',
    'ProteinMPNN+decode last',
    'RaSP','DDGun_dir','DDGun3D_dir', 'ACDC-NN-Seq_dir', 'ACDC-NN_dir']].rename(columns={c:c[:-4] for c in d.columns if c[-4:]=='_dir'}).rename(columns={'ACDC-NN':'ACDC-NN-Struct'})
  d['RaSP'] *= -1 # inkeeping with other predictors
  results['s669'] = {
    'all': compute_metrics(d, cols),
    'no_methionine': compute_metrics(d[~((d.pre=='M') | (d.post=='M'))], cols)
  }

  ### 2648
  f = f'{datasets_folder}/s2648/'
  paths = [
    f'{f}/esmif_s2648.csv',
    f'{f}/s2648_ddg_experimental.csv',
    f'{f}/proteinmpnn_s2648.csv',
    f'{f}/other_predictors_s2648.csv',
  ]
  d = read_and_join_csvs(paths)
  d = correct_esmif(d)
  d = d.rename(columns={f'prediction_DDGun{k}':f'DDGun{k}' for k in ('','3D')})
  d = d[['pdb_id', 'chain', 'pre', 'pos', 'post', 'ddG_experimental',
    'ESMif','ESMif-ddG',
    'ProteinMPNN', 'ProteinMPNN-ddG',
    'ProteinMPNN+decode last',
    'RaSP','DDGun','DDGun3D', 'ACDC-NN-Seq', 'ACDC-NN-Struct']]
  d['RaSP'] *= -1 # inkeeping with other predictors
  results['s2648'] = {
    'all': compute_metrics(d, cols),
    'no_methionine': compute_metrics(d[~((d.pre=='M') | (d.post=='M'))], cols)
  }

  print(results)

  # Latex for Tables 3 and 5:
  # Table 3: Accuracy of predictions for various models and datasets
  # Table 5: Results with all mutations involving methionine removed
  for k in ['all','no_methionine']:
    s = []

    for col in [
    'ProteinMPNN',
    'ProteinMPNN-ddG',
    'ESMif',
    'ESMif-ddG',
    'RaSP',
    'ACDC-NN-Seq',
    'ACDC-NN-Struct',
    'DDGun',
    'DDGun3D']:

      r = results['tsuboyama'][k].loc[col]
      lines = [    f'{col}',
        f"{r['top 10']*100:.0f}\%",
        # f"{r['auROC, identifying improving mutants']:.2f}",
        ]
      for dataset_name in ['tsuboyama','s2648','s669']:
        r = results[dataset_name][k].loc[col]
        lines+= [
          f"{r['roc_auc']:.2f}",
          f"{r['pr_auc']:.2f}",

          # f"{np.abs(r['spearman']):.2f}"
        ]

      s.append('{'+'} & {'.join(lines)+'} \\\\')

    # Hacky bolding and formatting (might be altered in final table in paper)
    s[1] = s[1].replace(' {',' \\textbf{{').replace('} ','}} ')
    s[1] = re.sub('}}', '}', s[1], 1)
    s.insert(0,r'{\textit{Unsupervised}}\\')
    s.insert(5,r'\midrule')
    s.insert(6,r'{\textit{Supervised}}\\')
    print('\n'.join(s)+'\n')

  # Latex for Table 4
  # Table 4: Ablation results for modifications of PROTEINMPNN
  for k in ['all']:
    s = []
    for col in [
    'ProteinMPNN',
    'ProteinMPNN+decode last',
    'ProteinMPNN-ddG'
    ]:

      r = results['tsuboyama'][k].loc[col]
      lines = [    f'{col}',
        f"{r['top 10']*100:.0f}\%",
        # f"{r['auROC, identifying improving mutants']:.2f}",
      ]
      for dataset_name in ['tsuboyama',
                # 's2648','s669'
                ]:
        r = results[dataset_name][k].loc[col]
        lines+= [
          f"{r['roc_auc']:.2f}",
          f"{r['pr_auc']:.2f}",

          # f"{np.abs(r['spearman']):.2f}"
        ]

      s.append('{'+'} & {'.join(lines)+'} \\\\')
    # Hacky bolding (might be altered in final table in paper)
    s[-1] = s[-1].replace(' {',' \\textbf{{').replace('} ','}} ')
    s[-1] = re.sub('}}', '}', s[-1], 1)
    s.insert(0,r'{\textit{Unsupervised}}\\')
    print('\n'.join(s)+'\n')
