import argparse
import pandas as pd
import numpy as np
import plotly.express as px

if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Plots a heatmap of the values and assymmetry in values of delta_X_to_Y averaged over single backbone geometries for ProteinMPNN')
  parser.add_argument("--coeff_path", help="Path to proteinmpnn coefficients", default='data/coeff_proteinmpnn_ddg_v_48_020.csv')
  parser.add_argument("--outfolder", default='data/', help='Path to write output plot pdf to, will write delta_X_Y.pdf and asymmetry_heatmap.pdf')
  args = parser.parse_args()

  outfolder = args.outfolder
  y = pd.read_csv(args.coeff_path, index_col=0)

  aas = list(np.abs(y+y.T).columns)
  for i in range(20):
    aa = aas[i:][np.sort(np.abs(y+y.T).loc[aas[i:]][aas[i:]].values, axis=1)[:,-3:].sum(1).argmax()]
    aas.remove(aa)
    aas.insert(0, aa)

  fig = px.imshow(np.abs(y+y.T).loc[aas[::-1]][aas[::-1]], color_continuous_scale='greys', template='none', height=550, width=500,
          range_color=[0,0.8]
  )
  fig.update_layout(xaxis_title='Amino acid', yaxis_title='Amino acid')
  fontsize = 13
  fig.update_layout(
    font=dict(
      size=fontsize,  # Set the font size here
    ),
  )
  fig.write_image(f'{outfolder}/asymmetry_heatmap.pdf','pdf')
  fig

  AA_FREQUENCY_LOGIT_CORRECTION = {
    'W': -1.368,
    'C': -1.260,
    'M': -0.944,
    'H': -0.569,
    'Y': -0.429,
    'F': -0.270,
    'Q': -0.206,
    'N': -0.156,
    'P': -0.098,
    'T': 0.058,
    'R': 0.087,
    'I': 0.102,
    'D': 0.122,
    'K': 0.230,
    'S': 0.264,
    'V': 0.280,
    'G': 0.310,
    'E': 0.362,
    'A': 0.448,
    'L': 0.630}
  aas = list(AA_FREQUENCY_LOGIT_CORRECTION.keys())

  fig = px.imshow(y.loc[aas[::-1]][aas],
            color_continuous_scale='RdBu',
            template='none', height=550, width=600,
            # range_color=[0,0.8]
            color_continuous_midpoint=0.,
          )
  fig.update_layout(xaxis_title='Mutant amino acid', yaxis_title='Wildtype amino acid')
  fontsize = 17
  fig.update_layout(
    font=dict(
      size=fontsize,  # Set the font size here
    ),
  )
  fig.write_image(f'{outfolder}/delta_X_Y.pdf','pdf')
  fig
