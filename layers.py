"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import CrossAttender, Encoder

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


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


# class RNNEncoder(nn.Module):
#     """General-purpose layer for encoding a sequence using a bidirectional RNN.

#     Encoded output is the RNN's hidden state at each position, which
#     has shape `(batch_size, seq_len, hidden_size * 2)`.

#     Args:
#         input_size (int): Size of a single timestep in the input.
#         hidden_size (int): Size of the RNN hidden state.
#         num_layers (int): Number of layers of RNN cells to use.
#         drop_prob (float): Probability of zero-ing out activations.
#     """
#     def __init__(self,
#                  input_size,
#                  hidden_size,
#                  num_layers,
#                  drop_prob=0.):
#         super(RNNEncoder, self).__init__()
#         self.drop_prob = drop_prob
#         self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
#                            batch_first=True,
#                            bidirectional=True,
#                            dropout=drop_prob if num_layers > 1 else 0.)

#     def forward(self, x, lengths):
#         # Save original padded length for use by pad_packed_sequence
#         orig_len = x.size(1)

#         # Sort by length and pack sequence for RNN
#         lengths, sort_idx = lengths.sort(0, descending=True)
#         x = x[sort_idx]     # (batch_size, seq_len, input_size)
#         x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

#         # Apply RNN
#         self.rnn.flatten_parameters()
#         x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

#         # Unpack and reverse sort
#         x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
#         _, unsort_idx = sort_idx.sort(0)
#         x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

#         # Apply dropout (RNN applies dropout after all but the last layer)
#         x = F.dropout(x, self.drop_prob, self.training)

#         return x


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
        self.att_linear_1 = nn.Linear(2 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

#         self.rnn = RNNEncoder(input_size=2 * hidden_size,
#                               hidden_size=hidden_size,
#                               num_layers=1,
#                               drop_prob=drop_prob)
        
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

        self.att_linear_2 = nn.Linear(2 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

        self.hidden_size = hidden_size
    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        #print (self.hidden_size, att.shape, mod.shape)
#         logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
#         mod_2 = self.rnn(mod, mask)
#         logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

#         # Shapes: (batch_size, seq_len)
#         log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
#         log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        
        logits_1 = self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask)
        logits_2 = self.mod_linear_2(mod_2)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)        

        return log_p1, log_p2
