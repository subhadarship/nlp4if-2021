import logging
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class SelfAttentionLayer(nn.Module):
    """Self attention layer class"""

    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        if hid_dim % n_heads != 0:
            raise AssertionError('hid_dim % n_head != 0')

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, seq len, seq len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, seq len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, seq len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, seq len, hid dim]

        x = self.fc(x)

        # x = [batch size, seq len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    """Position-wise feed forward layer class"""

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Encoder(nn.Module):
    """Encoder class"""

    def __init__(
            self,
            src_vocab_size,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            encoder_layer,
            self_attention_layer,
            positionwise_feedforward_layer,
            dropout,
            device,
    ):
        super().__init__()

        self.device = device
        self.hid_dim = hid_dim

        self.tok_embedding = nn.Embedding(src_vocab_size, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)  # max seq len in the data needs to be <=1000

        self.layers = nn.ModuleList(
            [encoder_layer(
                hid_dim,
                n_heads,
                pf_dim,
                self_attention_layer,
                positionwise_feedforward_layer,
                dropout,
                device
            ) for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask, return_attentions=False):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, src len]
        src = self.dropout(
            (self.tok_embedding(src) * self.scale) +
            self.pos_embedding(pos)
        )

        # src = [batch size, src len, hid dim]

        attentions = [] if return_attentions else None

        for layer in self.layers:
            src, attention = layer(src, src_mask)
            # attention = [batch size, n heads, src len, src len]
            if return_attentions:
                attentions.append(attention)

        # src = [batch size, src len, hid dim]

        # If return_attentions = True, attentions is a list containing n_layers tensors.
        # If return_attentions = False, attentions = None
        return src, attentions


class EncoderLayer(nn.Module):
    """Encoder layer class"""

    def __init__(
            self,
            hid_dim,
            n_heads,
            pf_dim,
            self_attention_layer,
            positionwise_feedforward_layer,
            dropout,
            device
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = self_attention_layer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = positionwise_feedforward_layer(
            hid_dim, pf_dim, dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, attention = self.self_attention(src, src, src, src_mask)

        # attention = [batch size, n heads, src len, src len]

        # dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward, dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(self.positionwise_feedforward(src)))

        # src = [batch size, src len, hid dim]

        return src, attention


class MultitaskTransformerEncoderClassificationModel(nn.Module):
    """Multitask Text classification model class using transformer encoder"""

    def __init__(
            self,
            encoder: Encoder,
            src_pad_idx: int,
            out_dims: List[int],
            fc_dim: int,
    ):
        super().__init__()

        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.out_dims = out_dims
        self.fc_dim = fc_dim
        hid_dim = self.encoder.hid_dim
        self.make_src_mask = self.make_src_mask_enc
        self.fc = nn.Linear(hid_dim, fc_dim)
        self.clfs = nn.ModuleList([nn.Linear(fc_dim, out_dim) for out_dim in self.out_dims])

    def make_src_mask_enc(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_src_mask_bert(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx)

        # src_mask = [batch size, src len]

        return src_mask.float()

    def forward(self, src, atts=False):
        # src = [batch size, src len]

        src_mask = self.make_src_mask(src)

        enc_src, attentions = self.encoder(src, src_mask, return_attentions=atts)

        # enc_src = [batch size, src len, hid dim]
        # attentions = [batch size, n heads, src len, src len]

        # only consider first token's representation
        enc_src = enc_src[:, 0, :]
        # enc_src = [batch size, hid dim]

        enc_src = self.fc(enc_src)
        # enc_src = [batch size, fc_dim]
        enc_src = F.relu(enc_src)
        # enc_src = [batch size, fc_dim]

        output = {f'q{idx + 1}': clf(enc_src) for idx, clf in enumerate(self.clfs)}
        # output = {q1: [batch size, out dim], ...}
        return {
            'logits': output,
            'attentions': attentions,
        }
