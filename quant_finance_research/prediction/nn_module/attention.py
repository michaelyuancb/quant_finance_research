import typing as ty
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F


class GateBlock(nn.Module):

    def __init__(self, input_dim: int, output_dim: int,
                 activate: ty.Optional[nn.Module] = None,
                 ):
        super(GateBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.activate = nn.Identity() if activate is None else activate

    def forward(self, x):
        x = self.linear(x)
        a, b = x.chunk(2, dim=-1)
        return a * self.activate(b)


class MultiheadAttention(nn.Module):
    """
    ref.
        [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
            Attention is all you need. Advances in neural information processing systems, 30.
        [2] Wang, Sinong, et al. "Linformer: Self-attention with linear complexity."
            arXiv preprint arXiv:2006.04768 (2020).
    """

    """
    Input:
        Universal Input:
        :param x_q Tensor(batch, n_token_q, dim_q)
        :param x_kv Tensor(batch, n_token_kv, dim_kv). if compression is None, n_token_q should be equal to n_token_kv.
        
        Linformer Input:
        :param key_compression: nn.Linear for compression in n_tokens dimension of key
        :param value_compression: nn.Linear for compression in n_tokens dimension of value, should be same property with
                key_compression for consistency of pair (key, value).
        Compression Mechanism: n_token Q  -----(query)----->  n_token_compression K & V.
    """

    def __init__(self, input_dim, n_heads, dropout=0.2,
                 qkv_dim=None,
                 initialization: str = 'kaiming'
                 ) -> None:
        super(MultiheadAttention, self).__init__()
        qkv_dim = input_dim if qkv_dim is None else qkv_dim
        self.W_q = nn.Linear(input_dim, qkv_dim)
        self.W_k = nn.Linear(input_dim, qkv_dim)
        self.W_v = nn.Linear(input_dim, qkv_dim)
        if n_heads > 1:
            assert qkv_dim % n_heads == 0
        assert initialization in ['xavier', 'kaiming']
        self.W_out = nn.Linear(qkv_dim, input_dim) if (qkv_dim != input_dim or n_heads > 1) else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return x.reshape(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2).reshape(batch_size * self.n_heads,
                                                                                             n_tokens, d_head)

    def forward(self, x_q: Tensor, x_kv: Tensor,
                # Linformer
                key_compression: ty.Optional[nn.Linear] = None,
                value_compression: ty.Optional[nn.Linear] = None,
                ) -> Tensor:

        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)  # (batch, ntk2, d_input)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # (batch, ntk2, d_input)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)  # (batch*n_heads, n_token_q, d_head)
        k = self._reshape(k)  # (batch*n_heads, ntk2, d_head)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)  # (batch*n_head, n_token_q, ntk2)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)  # (**, n_token_q, ntk2) * (**, ntk2, d_head)
        x = x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value).transpose(1, 2).reshape(
            batch_size, n_q_tokens, self.n_heads * d_head_value)  # (batch, n_token_q, n_head*d_head_value=d_value)
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    """
    ref.
        [1] https://github.com/kalelpark/FT_TransFormer
        [2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
            Attention is all you need. Advances in neural information processing systems, 30.
        [3] Wang, Sinong, et al. "Linformer: Self-attention with linear complexity."
            arXiv preprint arXiv:2006.04768 (2020).
        [4] Rubachev I, Alekberov A, Gorishniy Y, et al. Revisiting pretraining objectives for tabular deep learning[J].
            arXiv preprint arXiv:2207.03208, 2022.
    """

    """
    Universal Parameter:
    ::param input_dim: the dimension of each token.
    ::param n_layers:  the number of [MultiheadAttention + FFN] layer.
    ::param n_heads:   the number of multi-head.
    ::param output_dim: the dimension of output.
    ::param d_ffn_factor: the expansion coefficient for FFN hidden_dim, hidden_dim = d_ffn_factor * input_dim, Default=1
    ::param attention_dropout: the dropout rate of MultiheadAttention.
    ::param ffn_dropout: the dropout rate of FFN.
    ::param residual_dropout: the dropout rate of residual.
    ::param activation: the activate function, Default=nn.ReLU().
    ::param pre_normalization: bool, if True, layer-normalize before MHA & FFN, else normalize after MHA & FFN. Default=True.
    ::param initialization: the initialization algorithm to used, "kaiming" or "xavier", Default="kaiming".
    
    Linformer Parameter
    ::param kv_compression_rate: the compression-ratio, None if Linformer is not used.
    ::param n_tokens: the number of tokens, must be supported when Linformer is used.
    ::param kv_compression_rate: the Linformer-mode, Default='layerwise'. 
        Three mode are supported by the implementation:
        - 'layerwise' mode: all layer share one compression-layer.
        - 'headwise' mode:  all head share one compression-layer. (key and value share one compression).
        - 'key-value' mode: the key and value has their own compression-layer.
        
    Input
    ::input x: Tensor(batch, n_tokens, input_dim)
    ::input cls_mode: bool to decide whether to use cls mode in the last MultiheadAttention(MHA). Default=False.
            When cls_mode=True, the last MHA will only query with x[:, :1, :], then the output shape of MHA will 
            be Tensor(batch, 1, input_dim), which is used for final-downstream task, 
            and the output will be Tensor(batch, output_dim)
    
    Output
    ::output out: Tensor(batch, n_tokens, output_dim) is cls_mode=False. Else Tensor(batch, output_dim)
    """

    def __init__(self,
                 # transformer backbone
                 input_dim: int,
                 n_layers: int,
                 n_heads: int,
                 output_dim: int,
                 d_ffn_factor: float = 1.0,
                 attention_dropout: float = 0.2,
                 ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0,
                 activation: ty.Optional[nn.Module] = nn.ReLU(),
                 pre_normalization: bool = True,
                 initialization: str = "kaiming",
                 # linformer
                 kv_compression_rate: ty.Optional[float] = None,  # common recommendation: 0.25
                 n_tokens: int = None,
                 kv_compression_sharing: ty.Optional[str] = 'layerwise',
                 # str in ['layerwise', 'headwise', 'key-value']
                 ) -> None:
        if kv_compression_rate:
            assert kv_compression_sharing
            assert n_tokens
            assert kv_compression_sharing in ['layerwise', 'headwise', 'key-value']

        super(Transformer, self).__init__()
        self.activation = activation

        def make_kv_compression():
            assert kv_compression_rate
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression_rate), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_rate and kv_compression_sharing == 'layerwise'
            else None
        )

        d_hidden = int(input_dim * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        input_dim, n_heads, attention_dropout, initialization=initialization
                    ),
                    'gate_block': GateBlock(input_dim, d_hidden, self.activation),
                    'linear': nn.Linear(d_hidden, input_dim),
                    'norm1': nn.LayerNorm(input_dim),
                }
            )
            if not pre_normalization or layer_idx:
                layer['norm0'] = nn.LayerNorm(input_dim)
            if kv_compression_rate and self.shared_kv_compression is None:  # not 'layerwise'
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.pre_normalization = pre_normalization
        self.last_normalization = nn.LayerNorm(input_dim) if pre_normalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(input_dim, output_dim)

    def _get_kv_compressions(self, layer):
        if self.shared_kv_compression is not None:
            return self.shared_kv_compression, self.shared_kv_compression
        elif 'key_compression' in layer and 'value_compression' in layer:
            return layer['key_compression'], layer['value_compression']
        elif 'key_compression' in layer:
            return layer['key_compression'], layer['key_compression']
        else:
            return None, None

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.pre_normalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.pre_normalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x, cls_mode=False) -> Tensor:

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer and cls_mode else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['gate_block'](x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        if cls_mode:
            x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.activation(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    pass  # see debug file.
