import typing as ty

import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
import math
import torch.nn.init as nn_init

from quant_finance_research.prediction.embedding.basic_embedding import Basic_Embedding
from quant_finance_research.utils import get_example_cat_matrix


class CLS_Embedding(Basic_Embedding):
    """
        CLS Tokenizer: [CLS] + [Linear Transformation for Numerical] + [Embedding Book for Category]

        Notice that the input Tensor:(Batch, d_feat), the output embedding Tensor:(Batch, d_feat+1, d_token),
        with (Batch, 0, d_token) to be the cls for downstream task.

        ref.
        [1] Rubachev I, Alekberov A, Gorishniy Y, et al. Revisiting pretraining objectives for tabular deep learning[J].
            arXiv preprint arXiv:2207.03208, 2022.
    """
    """
    Parameter
    ::param d_feat: the dimension of input feature.
    ::param d_token: the dimension of token for each feature.
    ::param index_num: the list of index of numerical feature. Default=None
    ::param index_cat: the list of index of category feature. Default=None.
    ::param total_n_cat: the total num of the category. notice that this is a global concept. Default=None.
    ::param use_bias: whether to use a constant bias for token. Default=True.
    ::param use_index_recover: bool, whether to recover the order of index of embedding. Default=False.

    Input
    ::input x: the input Tensor(batch, d_feat)

    Output
    ::output token: the output Tensor(batch, d_feat+1, d_token), with Tensor(batch, 0, d_token) to be CLS Token.
    """
    def __init__(self, d_feat, d_token, index_num=None, index_cat=None, total_n_cat=None, use_bias=True,
                 use_index_recover=False) -> None:
        super().__init__(d_feat, d_token, index_num, index_cat, total_n_cat, use_index_recover)
        self.use_bias = use_bias

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(self.d_num + 1, d_token))
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if use_bias:
            self.bias = nn.Parameter(Tensor(self.d_feat+1, d_token))
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

        if index_cat is not None:
            self.category_embeddings = nn.Embedding(total_n_cat, d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

    def forward(self, x) -> Tensor:
        x = self.index_sort_func(x)
        x_num = torch.cat([torch.ones(len(x), 1, device=x.device)] +
                          [x[:, :self.d_num]] if self.d_num > 0 else [], dim=1)
        x_cls = self.weight[None] * x_num[:, :, None]
        if self.has_cat is not None:
            x_cls = torch.cat(
                [x_cls, self.category_embeddings(x[:, self.offset_cat:self.offset_cat+self.d_cat].int())],
                dim=1,
            )
        if self.bias is not None:
            x_cls = x_cls + self.bias
        if self.use_index_recover:
            x_cls = torch.concat([x_cls[:, 0:1], self.index_recover_func(x_cls[:, 1:])], dim=1)
        return x_cls


if __name__ == "__main__":
    x, y, ct_column = get_example_cat_matrix()
    tk = CLS_Embedding(d_feat=7, d_token=10, index_num=ct_column['index_num'],
                       index_cat=ct_column['index_cat'], total_n_cat=9, use_index_recover=True)
    x = torch.tensor(x)
    tkn = tk(x)
    print(tkn)
    print(tkn.shape)
