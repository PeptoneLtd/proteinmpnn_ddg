import argparse

import numpy as np
import pandas as pd

import jax
from jax import jit, vmap, grad, numpy as jnp
import plotly.express as px

ALPHABET = 'ARNDCQEGHILKMFPSTWYVX'
aas = np.array(ALPHABET)[None].view('U1')[:-1]
@vmap
def compute_angular_cosine(xyz):
  displacements = xyz[jnp.array([0,2])]-xyz[1]
  unit_vecs = displacements/jnp.linalg.norm(displacements, axis=-1, keepdims=True)
  dp = jnp.dot(*unit_vecs)
  return dp

@vmap
def compute_dihedrals(xyz):
  # assert xyz.shape == (4,3)
  d = xyz[1:]-xyz[:-1]
  index = jnp.arange(3)
  cross_pairs = jnp.stack([index[:-1], index[1:]])
  c = jnp.cross(*d[cross_pairs]) # c is the cross products of consecutive displacements
  p1 = jnp.einsum('i,i', d[0],c[1]) * jnp.linalg.norm(d[1])
  p2 = jnp.einsum('i,i', c[0],c[1])
  return jnp.arctan2(p1, p2)

def build_N_CA_C_opening_angle_plot(path_to_npz_structures):
  x = np.load(path_to_npz_structures)
  mask = (x['S']<20)
  S = x['S'][mask]

  cosines = compute_angular_cosine(x['X'][mask][...,:3,:])
  N_CA_C_angles = jnp.arccos(cosines)*180/jnp.pi
  kdes = {aa:
      jax.scipy.stats.gaussian_kde(N_CA_C_angles[S==i]) for i, aa in enumerate(aas)
        }

  x_vals = jnp.linspace(100,140,1000)
  vals = {aa: kdes[aa].evaluate(x_vals) for aa in aas}
  fig = px.line(pd.DataFrame(vals, index=x_vals), template='none', width=500, height=400, range_x=[102,122], labels={"variable": "Amino Acid"})
  color_dict = {k: px.colors.qualitative.D3[int(i)] for k,i in zip(aas, '77777771727777377774')}
  name_dict = {'V':'Valine','I':'Isoleucine','N':'Asparagine','P':'Proline','G':'Glycine'}
  fig.data = sorted(fig.data, key=lambda x: 'VIPG'.index(x['legendgroup']) if x['legendgroup'] in 'GPIV' else 0)
  for x in fig.data:
    aa = x['legendgroup']
    x['line']['color'] = color_dict[aa]
    x['line']['width'] = 3 if aa in 'GVIP' else 1.5
    x['showlegend'] = aa in 'GVIP'
    x['name'] = name_dict[aa] if aa in name_dict else ''
  fig.update_layout(xaxis_title='N-CA-C angle / degrees', yaxis_title='Probability density')

  fontsize = 16
  fig.update_layout(
      font=dict(
          size=fontsize,  # Set the font size here
      ),
  )
  fig.update_xaxes(automargin=True, showgrid=False, zeroline=False)
  fig.update_yaxes(automargin=True, showgrid=False, zeroline=False)
  fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.45,
    xanchor="right",
    x=1.03
  ))
  fig.update_xaxes(showline=True, linewidth=2, linecolor='black', ticks="inside")
  fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks="inside",dtick=0.1, range=[0,0.3])
  fig.update_layout(legend = dict(font = dict(size = fontsize)),
                    legend_title = dict(font = dict(size = fontsize)))
  return fig

def build_dihedrals_plot(path_to_npz_structures):
  x = np.load(path_to_npz_structures)

  mask = (x['S']<20)
  S = x['S'][mask]

  dihedrals = compute_dihedrals(x['X'][mask])
  N_CA_C_O_dihedrals = dihedrals*180/jnp.pi

  N_CA_C_O_dihedrals = jnp.where(N_CA_C_O_dihedrals<50, N_CA_C_O_dihedrals+180, N_CA_C_O_dihedrals)

  kdes = {aa:
    jax.scipy.stats.gaussian_kde(N_CA_C_O_dihedrals[S==i]) for i, aa in enumerate(aas)
  }

  x_vals = jnp.linspace(50,220,1000)
  vals = {aa: kdes[aa].evaluate(x_vals) for aa in aas}

  fig = px.line(pd.DataFrame(vals, index=x_vals), template='none', width=500, height=400, range_x=[85,210], labels={"variable": "Amino Acid"})
  color_dict = {k: px.colors.qualitative.D3[int(i)] for k,i in zip(aas, '77777771727777377774')}
  name_dict = {'V':'Valine','I':'Isoleucine','N':'Asparagine','P':'Proline','G':'Glycine'}
  fig.data = sorted(fig.data, key=lambda x: 'VIPG'.index(x['legendgroup']) if x['legendgroup'] in 'GPIV' else 0)
  for x in fig.data:
    aa = x['legendgroup']
    x['line']['color'] = color_dict[aa]
    x['line']['width'] = 3 if aa in 'GVIP' else 1.5
    x['showlegend'] = aa in 'GVIP'
    x['name'] = name_dict[aa] if aa in name_dict else ''
  fig.update_layout(xaxis_title='N-CA-C-O dihedral / degrees', yaxis_title='Probability density')

  fontsize = 16
  fig.update_layout(
    font=dict(
      size=fontsize,  # Set the font size here
    ),
  )
  fig.update_xaxes(automargin=True, showgrid=False, zeroline=False)
  fig.update_yaxes(automargin=True, showgrid=False, zeroline=False)
  fig.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.45,
    xanchor="right",
    x=1.03
  ))
  fig.update_xaxes(showline=True, linewidth=2, linecolor='black', ticks="inside")
  fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks="inside",dtick=0.01, range=[0,0.04])
  fig.update_layout(legend = dict(font = dict(size = fontsize)),
                    legend_title = dict(font = dict(size = fontsize)))
  return fig

if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Visualise backbone opening angles')
  parser.add_argument("--structure_data_path", help="Path of the .npz file of pre-filtered structure dataset which `compute_shifts.py` script generates", required=True)
  parser.add_argument("--outpath", help="Where to write pdf figure of N-Ca-C opening angle to", default='data/central_angles_distributions.pdf')
  args = parser.parse_args()

  path_to_npz_structures = args.structure_data_path # 'training_single_structure_per_cluster_23349_structures_5615050_residues.npz'
  fig = build_N_CA_C_opening_angle_plot(path_to_npz_structures)
  fig.write_image(args.outpath,'pdf')
