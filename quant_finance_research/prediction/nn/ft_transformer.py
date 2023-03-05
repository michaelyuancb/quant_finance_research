"""
    Model for FT_Transformer
    Reference:
    [1] https://github.com/kalelpark/FT_TransFormer
    [2] Rubachev I, Alekberov A, Gorishniy Y, et al. Revisiting pretraining objectives for tabular deep learning[J].
        arXiv preprint arXiv:2207.03208, 2022.
    [3] Wang, Sinong, et al. "Linformer: Self-attention with linear complexity." arXiv preprint arXiv:2006.04768 (2020).
    [3] Gorishniy Y, Rubachev I, Babenko A. On embeddings for numerical features in tabular deep learning[J].
        arXiv preprint arXiv:2203.05556, 2022.
"""

import numpy as np

import typing as ty
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch.nn as nn
from torch import Tensor
import torch.nn.init as nn_init
import torch
import torch.nn.functional as F
from quant_finance_research.prediction.embedding.cls_embedding import CLS_Embedding
from quant_finance_research.prediction.nn_module.attention import Transformer

ModuleType = Union[str, Callable[..., nn.Module]]


class FT_Transformer(nn.Module):
    """
    Embedding Tokenizer Parameter
    ::param d_feat: the dimension of input feature.
    ::param d_token: the dimension of token for each feature.
    ::param index_num: the list of index of numerical feature.
    ::param index_cat: the list of index of category feature.
    ::param total_n_cat: the total num of the category. notice that this is a global concept.

    Transformer Universal Parameter
    ::param n_layers:  the number of [MultiheadAttention + FFN] layer.
    ::param n_heads:   the number of multi-head.
    ::param output_dim: the dimension of output.

    Optional Parameter
    ::param use_token_bias: whether to use a constant bias for token in Embedding. Default=True.
    ::param use_index_recover: bool, whether to recover the order of index of embedding. Default=False
    ::param d_ffn_factor: the expansion coefficient for FFN hidden_dim, hidden_dim = d_ffn_factor * input_dim, Default=1
    ::param attention_dropout: the dropout rate of MultiheadAttention. Default=0.2
    ::param ffn_dropout: the dropout rate of FFN. Default=0.1
    ::param residual_dropout: the dropout rate of residual. Default=0.0
    ::param activation: the activate function, Default=nn.ReLU().
    ::param pre_normalization: bool, if True, layer-normalize before MHA & FFN, else normalize after MHA & FFN. Default=True.
    ::param initialization: the initialization algorithm to used, "kaiming" or "xavier", Default="kaiming".

    Linformer Parameter
    ::param kv_compression_rate: the compression-ratio, None if Linformer is not used.
    ::param kv_compression_rate: the Linformer-mode, Default='layerwise'.
        Three mode are supported by the implementation:
        - 'layerwise' mode: all layer share one compression-layer.
        - 'headwise' mode:  all head share one compression-layer. (key and value share one compression).
        - 'key-value' mode: the key and value has their own compression-layer.

    Input
    ::input x: Tensor(batch, input_dim)

    Output
    ::output out: Tensor(batch, n_tokens, output_dim) is cls_mode=False. Else Tensor(batch, output_dim)
    """

    def __init__(self,
                 # embedding tokenizer backbone
                 input_dim: int,
                 token_dim: int,
                 index_num: list,
                 index_cat: list,
                 total_n_cat: int,
                 # transformer backbone
                 n_layers: int,
                 n_heads: int,
                 output_dim: int,
                 # optional parameter
                 use_token_bias=True,
                 use_index_recover=False,
                 d_ffn_factor: float = 1.0,
                 attention_dropout: float = 0.2,
                 ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0,
                 activation: ty.Optional[nn.Module] = nn.ReLU(),
                 pre_normalization: bool = True,
                 initialization: str = "kaiming",
                 # linformer
                 kv_compression_rate: ty.Optional[float] = None,  # recommendation: 0.25 ?
                 kv_compression_sharing: ty.Optional[str] = 'layerwise',
                 # str in ['layerwise', 'headwise', 'key-value']
                 ) -> None:

        super(FT_Transformer, self).__init__()
        self.tokenizer = CLS_Embedding(d_feat=input_dim, d_token=token_dim, index_num=index_num, index_cat=index_cat,
                                       total_n_cat=total_n_cat, use_bias=use_token_bias,
                                       use_index_recover=use_index_recover)
        n_tokens = self.tokenizer.d_feat + 1  # cls for extra one.
        self.transformer = Transformer(input_dim=token_dim, n_layers=n_layers, n_heads=n_heads,
                                       d_ffn_factor=d_ffn_factor, output_dim=output_dim,
                                       attention_dropout=attention_dropout,
                                       ffn_dropout=ffn_dropout, residual_dropout=residual_dropout,
                                       activation=activation, pre_normalization=pre_normalization,
                                       initialization=initialization,
                                       kv_compression_rate=kv_compression_rate,
                                       n_tokens=n_tokens,
                                       kv_compression_sharing=kv_compression_sharing)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tokenizer(x)
        x = self.transformer(x, cls_mode=True)
        return x


from quant_finance_research.utils import get_example_cat_df
from quant_finance_research.eda.eda_feat import eda_data_type
from quant_finance_research.prediction.embedding.embedding_utils import category2int_global

if __name__ == "__main__":
    # Examples of Test CODE:
    #     >> Test CODE
    df, df_column = get_example_cat_df()
    dict_abs, dict_rel, result_df = eda_data_type(df, df_column)
    df_cat_pro, total_n_cat = category2int_global(df, df_column, dict_abs['x_object'] + dict_abs['x_int'])
    print(dict_abs)
    print(len(df_column['x']))
    ft_transformer = FT_Transformer(
        input_dim=len(df_column['x']),
        token_dim=16,
        index_num=dict_rel['x_real'],
        index_cat=dict_rel['x_object']+dict_rel['x_int'],
        total_n_cat=total_n_cat,
        n_layers=3, n_heads=2, output_dim=2,
        activation=nn.GELU(),
        kv_compression_rate=0.5,
        kv_compression_sharing='key-value'
    )
    X = torch.tensor(df_cat_pro.iloc[:, df_column['x']].values).float()
    pred = ft_transformer(X)
    print(pred)
    print(pred.shape)
