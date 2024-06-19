import argparse
from time import time

import numpy as np
import pandas as pd
import plotly.express as px
import jax
from jax import jit, vmap, numpy as jnp

from proteinmpnn_ddg import get_compute_logit_differences_fn, RunModel, _aa_convert, mk_mpnn_model

def get_compute_logit_differences_fn_slow(model, scale_by_nres=True, batched=False):
  run_model = RunModel(model._model.config)
  run_model.params = model._model.params
  @jit
  def _compute_logit_differences(I, key):
    nres = I['S'].shape[0]
    nrepeats = nres if batched else (min(nres, 65536//nres) if (scale_by_nres) else 1)
    I['S'] = _aa_convert(I['S']).argmax(-1)
    I['decoding_order'] = jax.random.uniform(key, (nrepeats,nres)).argsort(1)
    predict_fn = vmap(run_model.score, in_axes=(None,None,{
      k:0 if k=='decoding_order' else None for k in I})) if (not batched) else run_model.batched_random_order_score
    o = predict_fn(run_model.params, key, I)
    logit_differences = jax.vmap(lambda x,y: x-x[y])(o['logits'][(jnp.diag_indices(nres) if scale_by_nres else 0)],o['S'][0])
    return _aa_convert(logit_differences, rev=True)
  return _compute_logit_differences

def benchmark(outfolder, n):
  model = mk_mpnn_model('v_48_020')
  score_fast = get_compute_logit_differences_fn(model)
  score_slow = get_compute_logit_differences_fn_slow(model, scale_by_nres=True)
  score_single = get_compute_logit_differences_fn_slow(model, scale_by_nres=False)
  score_slow_batched = get_compute_logit_differences_fn_slow(model, batched=True)

  key = jax.random.PRNGKey(0)
  I = jax.device_put({
    'X': jax.random.uniform(key, (n,4,3))*10,
    'S': np.zeros(n, dtype=int),
    'residue_idx': np.arange(n, dtype=int)+1,
    'mask': np.ones(n, dtype=float),
    'chain_idx': np.zeros(n, dtype=int),
  })

  def benchmark(f, inputs, nrepeats=10):
    jax.block_until_ready(f(*inputs)) # compile
    start = time()
    for _ in range(nrepeats):
      jax.block_until_ready(f(*inputs))
    end = time()
    return (end-start)/nrepeats

  timings = {}
  for n in (2**np.arange(5,int(np.log2(n))+1))[::-1]:
    print(n)
    I = {k:v[:n] for k,v in I.items() if k in ['S', 'X', 'chain_idx', 'decoding_order', 'mask', 'residue_idx']}
    timings[n]={}
    timings[n]['shared_orders'] =  benchmark(score_fast, (I, key))
    timings[n]['separate_orders_batched'] =  benchmark(score_slow_batched, (I, key), nrepeats=3)
    timings[n]['separate_orders'] =  benchmark(score_slow, (I, key), nrepeats=3)* (n/min(n, 65536//n))
    timings[n]['single_order'] =  benchmark(score_single, (I, key))

  d = pd.DataFrame(timings).T

  d['slowdown_shared_vs_single'] = d['shared_orders']/d['single_order']
  # d['speedup_shared_vs_separate'] = d['separate_orders_batched']/d['shared_orders']
  d['slowdown_separate_vs_single'] = d['separate_orders_batched']/d['single_order']

  # n = d.index.values
  # d['theoretical_max_slowdown_shared'] = np.log2(n).astype(int)+1
  # d['theoretical_max_slowdown_separate'] = n

  print(d.T)
  d.to_csv(f'{outfolder}/timings_benchmark.csv', index=False)

  fig = px.line(d[['slowdown_separate_vs_single','slowdown_shared_vs_single']], log_y=True, template='none', markers=True,  labels={"variable": "Implementation"})
  fig.update_xaxes(type='log', dtick=0.30102999566, title='Number of Residues', showgrid=False, zeroline=False)
  fig.update_yaxes(title='Slowdown relative to single pass')
  fig.update_layout(width=600, height=500, showlegend=True)
  fontsize = 16
  fig.update_layout(
    font=dict(
      size=fontsize,  # Set the font size here
    ),
  )
  fig.update_traces(marker=dict(size=fontsize,),
            line=dict(width=fontsize*0.3),
  )
  fig.update_yaxes(automargin=True, showgrid=False, zeroline=False)
  fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=.95
  ))
  newnames = {'slowdown_shared_vs_single':'Tied decoding', 'slowdown_separate_vs_single': 'Naive'}
  fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                      legendgroup = newnames[t.name],
                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                    )
            )
  for x in fig.data:
    x['showlegend'] = False

    fig.add_shape(
      # legendwidth=70,
      name  = x['name'],
      legendgroup = x['name'],
      showlegend = True,
      type = 'circle',
      layer = 'below',
      line = dict(width = 0),
      fillcolor = x['line']['color'],
      x0 = x['x'][0],
      y0 = x['y'][0],
      x1 = x['x'][0],
      y1 = x['y'][0],
      # marker=dict(size=24),
    )
  fig.update_xaxes(showline=True, linewidth=2, linecolor='black', ticks="inside")
  fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks="inside")
  fig.update_yaxes(range=[0., 3.2])
  fig.update_layout(legend = dict(font = dict(size = fontsize)),
            legend_title = dict(font = dict(size = fontsize)))
  fig.write_image(f'{outfolder}/timings_benchmark.pdf','pdf')

if (__name__=='__main__'):
  parser = argparse.ArgumentParser(description='Benchmark speed of ProteinMPNN-ddG predictions with various implementations')
  parser.add_argument("--outfolder", default='data/', help='Folder to write output to, will write timings_benchmark.csv and timings_benchmark.pdf')
  parser.add_argument("--n", default=4096, type=int, help='Size of max protein size to predict')
  args = parser.parse_args()
  benchmark(args.outfolder, args.n)
