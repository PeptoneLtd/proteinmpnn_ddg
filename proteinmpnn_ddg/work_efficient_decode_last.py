import jax
from jax import vmap, jit, numpy as jnp
import haiku as hk
from colabdesign.mpnn.utils import cat_neighbors_nodes, get_ar_mask
from colabdesign.mpnn.modules import ProteinMPNN, RunModel
from colabdesign.mpnn.model import _aa_convert

# compute_depth = lambda x: jnp.where(x.all(), x.shape[0]-1, jnp.where(~x[0], 0, (jnp.diff(x.cumsum())==1).argmin()))
def compute_depth(bits):
  same_bit_bool = (bits[:,None]==bits[None,:]).reshape(-1, bits.shape[-1])
  def _body(carry, vals):
    (all_prev_true, level) = carry
    all_prev_true &= vals
    level+=all_prev_true
    return (all_prev_true, level), None
  (_, level), _ = jax.lax.scan(
    _body,
    init=(
      jnp.ones(same_bit_bool.shape[0], dtype=bool),
      jnp.zeros(same_bit_bool.shape[0], dtype=int)-1
    ),
    xs=same_bit_bool.T[::-1],
  )
  return level.reshape(bits.shape[:1]*2)

def compute_levels(nt):
  n_levels = jnp.ceil(jnp.log2(nt)).astype(int)+1
  bits = jnp.unpackbits(
    jnp.arange(nt, dtype=jnp.uint32).view(jnp.uint8),
    bitorder='little'
    ).reshape(-1,32)[:,:n_levels].astype(bool)
  levels = compute_depth(bits)
  return levels, n_levels

class ProteinMPNN(ProteinMPNN):
  def order_agnostic_score_preface(self, I):
    """
    Structure encoder, independent of decoding order as fully visible

    I = {
         [[required]]
         'X' = (L,4,3)
         'mask' = (L,)
         'residue_index' = (L,)
         'chain_idx' = (L,)

         [[optional]]
         'S' = (L,21)
        }
    """
    key = hk.next_rng_key()
    # Prepare node and edge embeddings
    E, E_idx = self.features(I)
    h_V = jnp.zeros((E.shape[0], E.shape[-1]))
    h_E = self.W_e(E)
    # Encoder is unmasked self-attention
    mask_attend = jnp.take_along_axis(I["mask"][:,None] * I["mask"][None,:], E_idx, 1)
    for layer in self.encoder_layers:
      h_V, h_E = layer(h_V, h_E, E_idx, I["mask"], mask_attend)
    # Build encoder embeddings
    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    ##########################################
    # conditional_probs - only
    ##########################################

    # Concatenate sequence embeddings for autoregressive decoder
    h_S = self.W_s(I["S"])
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    return h_V, h_EXV_encoder, h_ES, E_idx

  def order_dependent_score_level(self, i, E_idx_levels, I, h_Vs_l, h_EXV_encoder, h_ES, E_idx, mask_attend):
    mask_1D = I['mask'][:,None]
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1 - mask_attend)
    h_EXV_encoder_fw = mask_fw[...,None] * h_EXV_encoder

    for layer_index, layer in enumerate(self.decoder_layers):
      h_ESV = jnp.concatenate([
        h_ES,
        vmap(lambda x,y: x[y], in_axes=(None,0))(h_Vs_l[:,layer_index], (E_idx_levels, E_idx))
      ], axis=-1)
      h_ESV = mask_bw[...,None] * h_ESV + h_EXV_encoder_fw
      h_Vs_l = h_Vs_l.at[i, layer_index+1].set(layer(h_Vs_l[i, layer_index], h_ESV, I['mask']))
    return h_Vs_l

  def decode_h_V(self, h_V):
    logits = self.W_out(h_V)
    return logits

  def order_dependent_score(self, I, h_V, h_EXV_encoder, h_ES, E_idx):
    # While decoding is always last, the tied orders up to then can be varied
    # that varying is done in decoding order. As structure encoder is not
    # autoregressive (fully visible) we can use a shuffle
    if 'decoding_order' in I:
      order = I['decoding_order']
      # shuffle according to decoding order
      I = {k: v[order] if (k!='decoding_order') else v for k,v in I.items() }
      h_V, h_EXV_encoder, h_ES, E_idx = (v[order] for v in (h_V, h_EXV_encoder, h_ES, E_idx))
      E_idx = order.argsort()[E_idx]

    h_Vs = jnp.stack([h_V]+[jnp.empty_like(h_V)]*3) # start, and then 1 for each layer output in decoder
    nt = I['S'].shape[0]
    with jax.ensure_compile_time_eval():
      # The tied decoding plan
      levels, n_levels = compute_levels(nt)
    h_Vs_l = jnp.tile(h_Vs[None], (n_levels,)+(1,)*(h_Vs.ndim)) # (n_levels, n_layers=4, n_residues, hidden_dim=128)

    def process_level(i, h_Vs_l):
      ar_mask = jnp.tril((levels!=i), k=-1) + (levels<=i-1)
      mask_attend = jnp.take_along_axis(ar_mask, E_idx, 1)
      E_idx_levels = jnp.take_along_axis(jnp.clip(levels, a_max=i), E_idx, axis=1)
      h_Vs_l = self.order_dependent_score_level(i, E_idx_levels, I, h_Vs_l, h_EXV_encoder, h_ES, E_idx, mask_attend)
      return h_Vs_l

    h_Vs_l = jax.lax.fori_loop(0, n_levels, process_level, h_Vs_l)
    logits = self.decode_h_V(h_Vs_l[-1,-1])
    S = I.get("S", None)

    if 'decoding_order' in I:
      inverse_order = order.argsort()
      logits, S = (x[inverse_order] for x in (logits, S)) # reverse the shuffling
    return {"logits": logits, "S":S}

  def work_efficient_decode_last(self, I):
    h_V, h_EXV_encoder, h_ES, E_idx = self.order_agnostic_score_preface(I)
    return self.order_dependent_score(I, h_V, h_EXV_encoder, h_ES, E_idx)

  def work_efficient_decode_last_vmap_decoding_order(self, I):
    '''
    Function to decode all tokens last, but with a batch of different
    underlying tied decoding orders up until that final residue
    '''
    h_V, h_EXV_encoder, h_ES, E_idx = self.order_agnostic_score_preface(I)
    o = vmap(
      self.order_dependent_score,
      in_axes=({
        k: None if (k!='decoding_order') else 0 for k in I.keys()
        },)+(None,)*4
    )(I, h_V, h_EXV_encoder, h_ES, E_idx)
    o['S'] = o['S'][0]
    return o

  def random_single_order_decoding(self, I, h_V, h_EXV_encoder, h_ES, E_idx):
    '''
    Normal ProteinMPNN code, decoding with a particular order
    '''
    # get autoregressive mask
    if "ar_mask" in I:
      decoding_order = None
      ar_mask = I["ar_mask"]
    else:
      decoding_order = I["decoding_order"]
      ar_mask = get_ar_mask(decoding_order)

    mask_attend = jnp.take_along_axis(ar_mask, E_idx, 1)
    mask_1D = I["mask"][:,None]
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1 - mask_attend)

    h_EXV_encoder_fw = mask_fw[...,None] * h_EXV_encoder
    for layer in self.decoder_layers:
      # Masked positions attend to encoder information, unmasked see.
      h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
      h_ESV = mask_bw[...,None] * h_ESV + h_EXV_encoder_fw
      h_V = layer(h_V, h_ESV, I["mask"])

    logits = self.W_out(h_V)
    S = I.get("S",None)
    return {"logits": logits, "decoding_order":decoding_order, "S":S}

  def batched_random_order_score(self, I, meta_batch_size=65536):
    '''
    Normal ProteinMPNN, but with a batch of decoding orders for the
    same protein. Minibatched to avoid OOM.
    '''
    nres = I['S'].shape[0]
    n_decodings = I['decoding_order'].shape[0]
    batch_size = min(n_decodings, meta_batch_size//nres)
    assert (n_decodings%batch_size==0), "Minibatch size must evenly divide number of decoding orders"
    h_V, h_EXV_encoder, h_ES, E_idx = self.order_agnostic_score_preface(I)
    decoding_orders = I['decoding_order']
    @vmap
    def _decode_fn(decoding_order):
      return self.random_single_order_decoding({'decoding_order': decoding_order, **{k:I[k] for k in ['X','S','residue_idx','chain_idx','mask']}}, h_V, h_EXV_encoder, h_ES, E_idx)
    def _body(_, decoding_order):
      return None, _decode_fn(decoding_order)
    _, outputs = jax.lax.scan(
      _body,
      init = None,
      xs = decoding_orders.reshape(-1, batch_size, nres),
    )
    return jax.tree.map(lambda x: x.reshape((-1,)+x.shape[2:]), outputs)

class RunModel(RunModel):
  def __init__(self, config) -> None:
    self.config = config
    # expose functions from haiku
    def _transform(f_name):
      def _f(*args, **kwargs):
        model = ProteinMPNN(**self.config)
        return getattr(model, f_name)(*args, **kwargs)
      return _f
    for k in [
      'score',
      'sample',
      'order_agnostic_score_preface',
      'decode_h_V',
      'order_dependent_score_level',
      'order_dependent_score',
      'work_efficient_decode_last',
      'work_efficient_decode_last_vmap_decoding_order',
      'batched_random_order_score'
      ]:
      setattr(self, k,  hk.transform(_transform(k)).apply)

def get_compute_logit_differences_fn(model, work_efficient=True):
  run_model = RunModel(model._model.config)
  run_model.params = model._model.params
  @jit
  def _compute_logit_differences(I, key):
    I['S'] = _aa_convert(I['S']).argmax(-1)
    I['decoding_order'] = jax.random.uniform(key, I['S'].shape).argsort()
    _predict_fn = getattr(run_model,
      'work_efficient_decode_last' if work_efficient else 'score'
    )
    o = _predict_fn(run_model.params, key, I)
    logit_differences = jax.vmap(lambda x,y: x-x[y])(o['logits'], o['S'])
    return _aa_convert(logit_differences, rev=True)
  return _compute_logit_differences
