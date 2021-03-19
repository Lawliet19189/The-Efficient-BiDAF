"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
from entmax import entmax15
from collections import namedtuple
import torch.nn.functional as F

from einops import rearrange
from torch import einsum
from util import masked_softmax
from util import groupby_prefix_and_trim, exists, default, equals, max_neg_value
from functools import partial

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors, freeze=False)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.char_proj = nn.Linear(char_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, 2 * hidden_size) # 2*H due to concatination of char and word embeddings

    def forward(self, x, y):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)


        char_emb = self.char_embed(y)
        char_emb = torch.mean(char_emb, -2)
        char_emb = F.dropout(char_emb, self.drop_prob, self.training)
        char_emb = self.char_proj(char_emb)
        
        concat_emb = torch.cat((char_emb, emb), 2)
        concat_emb = self.hwy(concat_emb)
        return concat_emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class TBiDAFAttention(nn.Module):
    def __init__(self, hidden_size, drop_prob=0.2):
        super(TBiDAFAttention, self).__init__()
        self.att = CrossAttender(dim=hidden_size,
                                 depth=6,
                                 heads=12,
                                 #sandwich_coef=2,
                                 #residual_attn=True,
                                 #attn_num_mem_kv=16,
                                 ff_glu=True,
                                 rel_pos_bias=False,
                                 dropout=0.1,
                                 position_infused_attn=True,
                                 #cross_attend=True,
                                 #only_cross=True,
                                 use_scalenorm=True
                                 )

    def forward(self, c, q, c_mask, q_mask):
        att = self.att(c, context=q, mask=c_mask, context_mask=q_mask)  # (batch_size,
        # c_len, hidden_size)
        
        x = torch.cat([c, att], dim=2)  # (bs, c_len, 2 * hid_size)
        #print (c.shape, q.shape, att.shape, x.shape)
        return x


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)
        
        self.rnn = Encoder(
            dim=2*hidden_size,
            depth=1,
            heads=8,
            ff_glu=True,
            ff_dropout=drop_prob,
            attn_dropout=drop_prob,
            use_scalenorm=True,
            position_infused_attn=True
        )
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

        self.hidden_size = hidden_size

    def forward(self, att, mod, mask):

        logits_1 = self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask)
        logits_2 = self.mod_linear_2(mod_2)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)        

        return log_p1, log_p2


# Transformer layers

Intermediates = namedtuple(
    'Intermediates', [
        'pre_softmax_attn',
        'post_softmax_attn'
    ]
)

LayerIntermediates = namedtuple(
    'Intermediates', [
        'hiddens',
        'attn_intermediates'
    ]
)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :] # -> (1, batch_size, max_seq_len)


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1./ (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + \
            offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g


class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


class GEGLU(nn.Module):  # adopted from paper: https://arxiv.org/abs/2002.05202
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out*2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=True, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            causal=False,
            mask=None,
            attn_use_entmax15=True,
            num_mem_kv=0,
            dropout=0.2,
            on_attn=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        inner_dim = dim * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.attn_fn = entmax15

        self.num_mem_kv = num_mem_kv

        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim*2),
            nn.GLU()
        ) if on_attn else nn.Linear(
            inner_dim, dim
        )

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            rel_pos=None,
            sinusoidal_emb=None,
            prev_attn=None,
            mem=None
    ):
        b, n, _, h, device = *x.shape, self.heads, x.device
        kv_input = default(context, x)  # For Cross-Attention

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            offset = k_input.shape[-2] - q_input.shape[-2] # for cross-attention,
            # difference in seq length
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        # compute each head embedding from number_of_heads * y = embed_size
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]),
                                                        device=device).bool())

            # Create like context mask * context mask matrix with each value in the
            # matrix representing whether to compute attention for that or not.
            q_mask = rearrange(q_mask, 'b i -> b () i ()')  # (B, N) -> (B, 1, N, 1)
            k_mask = rearrange(k_mask, 'b j -> b () () j')  # (B, M) -> (B, 1, 1, M)
            input_mask = q_mask * k_mask # (B, 1, N, 1) * (B, 1, 1, M) -> (B, 1, N, M)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots += prev_attn

        pre_softmax_attn = dots

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        intermediates = Intermediates(
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return self.to_out(out), intermediates

class AttentionLayers(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads = 8,
            cross_attend = False,
            only_cross = False,
            position_infused_attn = True,
            custom_layers = None,
            sandwich_coef = None,
            residual_attn = False,
            pre_norm = True,
            causal = False,
            use_scalenorm = True,
            #dim_out=None,
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.dim_out=dim

        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else \
            None

        # To do: Implement relative position bias
        # To do: Implement residual attention

        self.pre_norm = pre_norm

        # ScaleNorm from paper: https://arxiv.org/abs/1910.05895
        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_fn = partial(norm_class, dim)

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')  # Attention -> cross -> Feedforward
        elif cross_attend and only_cross:
            default_block = ('c', 'f')  # Cross -> Feedforward
        else:
            default_block = ('a', 'f')  # Attention -> Feedforward

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(sandwich_coef):
            # TO DO
            pass
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer,
                Residual()
            ]))

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            mems=None,
            return_hiddens=False
    ):
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        for idx, (layer_type, (norm, block, residual_fn)) in enumerate(zip(
                self.layer_types, self.layers)):
            is_last = idx == (len(self.layers) - 1)

            if layer_type == "a":
                hiddens.append(x)
                layer_mem = mems.pop(0)

            residual = x

            if self.pre_norm:
                x = norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask=mask, sinusoidal_emb=self.pia_pos_emb,
                                   rel_pos=None, prev_attn=prev_attn, mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, sinusoidal_emb=self.pia_pos_emb,
                                   context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if not self.pre_norm and not is_last:
                x = norm(x)

            if return_hiddens:
                intermediates = LayerIntermediates(
                    hiddens=hiddens,
                    attn_intermediates=intermediates
                )

                return x, intermediates

            return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)


class TransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            max_seq_len,
            attn_layers,
            emb_dim=None,
            emb_dropout = 0.,
            num_memory_tokens = None,
            use_pos_emb = True,
            word_vectors = None,
            char_vectors = None,
            hidden_size = None,
            drop_prob = 0.,
    ):
        super().__init__()

        dim = attn_layers.dim # encoder dimension
        self.max_seq_len = max_seq_len
        self.num_memory_tokens = 0
        self.token_emb = Embedding(word_vectors, char_vectors, hidden_size, emb_dropout)
        if use_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
        else:
            self.pos_emb = FixedPositionalEmbedding(emb_dim)

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        # To DO: Add Memory tokens from paper https://arxiv.org/abs/2006.11527

    def forward(
            self,
            x, # -> (word_idxs, char_idxs)
            mask = None,
            **kwargs
    ):
        x = self.token_emb(x[0], x[1])
        x += self.pos_emb(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        b, n, device, num_mem = x.shape[0], x.shape[1], x.device, self.num_memory_tokens

        # To DO:
        if num_mem > 0:
            pass

        x, intermediates = self.attn_layers(x, mask=mask, return_hiddens = True, **kwargs)
        x = self.norm(x)

        return x
