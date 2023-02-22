import typing as ty
import math

import torch
from datetime import datetime
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import torch.nn.functional as F

from quant_finance_research.prediction.nn_module.attention import *


class DebugMultiheadAttention:

    def __init__(self):
        self.q = torch.randn(64, 10, 60)
        self.kv = torch.randn(64, 8, 60)
        self.mhat = MultiheadAttention(60, 4, dropout=0.1, qkv_dim=40)
        self.linear_k = nn.Linear(8, 6)
        self.linear_v = nn.Linear(8, 6)

    def debug_forward(self):
        y = self.mhat(self.q, self.kv, key_compression=self.linear_k, value_compression=self.linear_v)
        print(y)
        print(y.shape)


class DebugTransformer:

    def __init__(self):
        input_dim = 48
        n_tokens = 100
        batch = 4
        self.token = torch.randn(batch, n_tokens, input_dim)
        self.transformer_1 = Transformer(input_dim=input_dim, n_tokens=n_tokens, n_layers=3, n_heads=4, d_ffn_factor=2,
                                         output_dim=2)
        self.transformer_2 = Transformer(input_dim=input_dim, n_tokens=n_tokens, n_layers=3, n_heads=4, d_ffn_factor=2,
                                         output_dim=2,
                                         pre_normalization=True)
        self.linformer_1 = Transformer(input_dim=input_dim, n_layers=3, n_heads=4, d_ffn_factor=2, output_dim=2,
                                       kv_compression_rate=0.25, n_tokens=n_tokens, kv_compression_sharing='layerwise')
        self.linformer_2 = Transformer(input_dim=input_dim, n_layers=3, n_heads=4, d_ffn_factor=2, output_dim=2,
                                       kv_compression_rate=0.25, n_tokens=n_tokens, kv_compression_sharing='headwise')
        self.linformer_3 = Transformer(input_dim=input_dim, n_layers=3, n_heads=4, d_ffn_factor=2, output_dim=2,
                                       kv_compression_rate=0.25, n_tokens=n_tokens, kv_compression_sharing='key-value')

    def debug_forward(self):
        now = datetime.now()
        out_1 = self.transformer_1(self.token)
        print(f"transformer_cost={datetime.now()-now}")
        print(out_1.shape)
        out_2 = self.transformer_2(self.token)
        print(out_2.shape)
        now = datetime.now()
        out_3 = self.linformer_1(self.token)
        print(f"linformer_cost={datetime.now()-now}")
        print(out_3.shape)
        out_4 = self.linformer_2(self.token)
        print(out_4.shape)
        out_5 = self.linformer_3(self.token)
        print(out_5.shape)
        out_1 = self.transformer_1(self.token, cls_mode=True)
        print(out_1.shape)
        out_5 = self.linformer_3(self.token, cls_mode=True)
        print(out_5.shape)


if __name__ == "__main__":
    # DebugMultiheadAttention().debug_forward()
    DebugTransformer().debug_forward()
