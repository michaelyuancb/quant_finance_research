from torch import nn
from quant_finance_research.utils import get_example_cat_matrix
import torch


class Basic_Embedding(nn.Module):
    """
        Basic Embedding, which means do nothing but reshape.
        Need to be inherited by other Tokenizer in QuantFinanceResearch.
    """
    """
    Parameter
    ::param d_feat: the dimension of input feature.
    ::param d_token: the dimension of token for each feature.
    ::param index_num: the list of index of numerical feature. Default=None
    ::param index_cat: the list of index of category feature. Default=None.
    ::param total_n_cat: the total num of the category. notice that this is a global concept. Default=None.
    ::param use_index_recover: bool, whether to recover the order of index of embedding. Default=False.
    
    Input
    ::input x: the input Tensor(batch, d_feat)
    
    Output
    ::output token: the output Tensor(batch, d_feat, d_token)
    """

    def __init__(self, d_feat, d_token, index_num=None, index_cat=None, total_n_cat: int=None,
                 use_index_recover: bool=False):
        super(Basic_Embedding, self).__init__()

        if type(index_num) is not list:
            index_num = list(index_num)
        if type(index_cat) is not list:
            index_cat = list(index_cat)
        if index_cat is not None and len(index_cat) > 0 and total_n_cat is None:
            raise ValueError(f"When there is category, the total cat_amount should not be None.")
        self.total_n_cat = total_n_cat
        self.index_num = index_num if index_num is not None else []
        self.index_cat = index_cat if index_cat is not None else []
        self.d_num = len(index_num) if index_num is not None else 0
        self.d_cat = len(index_cat) if index_cat is not None else 0
        idx_list = self.index_num + self.index_cat
        idx_list = list(set(idx_list))
        d_feat_calc = len(idx_list)
        if d_feat != d_feat_calc:
            raise ValueError(f"The total number of index = {d_feat_calc}, which is not eqaul to d_feat={d_feat}")
        self.d_feat = d_feat
        self.d_token = d_token
        self.has_cat = 1 if self.d_cat > 0 else None
        self.has_num = 1 if self.d_num > 0 else None
        self.offset_num = 0
        self.offset_cat = self.d_num
        self._sort_index = self.index_num + self.index_cat
        _recv_idx = [0] * d_feat
        for i in range(self.d_num):
            _recv_idx[self.index_num[i]] = self.offset_num+i
        for i in range(self.d_cat):
            _recv_idx[self.index_cat[i]] = self.offset_cat+i
        self._recv_index = _recv_idx
        self.use_index_recover = use_index_recover

    def index_sort_func(self, x):
        x = x[:, self._sort_index]
        return x

    def index_recover_func(self, x):
        x = x[:, self._recv_index]
        return x

    def forward(self, x):
        x = self.index_sort_func(x)
        batch = x.shape[0]
        x = x.view(batch, -1, 1)
        if self.use_index_recover:
            x = self.index_recover_func(x)
        return x

    def get_offset_num(self):
        return self.offset_num

    def get_offset_cat(self):
        return self.offset_cat

    def get_offset(self):
        return {"num": self.get_offset_num(), "cat": self.get_offset_cat()}

    def get_d_feat(self):
        return self.d_feat

    def get_d_token(self):
        return self.d_token

    def get_d_num(self):
        return self.d_num

    def get_d_cat(self):
        return self.d_cat

    def get_d(self):
        return {"feat": self.d_feat, "num": self.get_d_num(), "cat": self.get_d_cat()}

    def get_index_num(self):
        return self.index_num

    def get_index_cat(self):
        return self.index_cat

    def get_index(self):
        return {"num": self.get_index_num(), "cat": self.get_index_cat()}


if __name__ == "__main__":
    x, y, index = get_example_cat_matrix()
    x = torch.tensor(x)
    print(x.shape)
    tokenizer = Basic_Embedding(7, 0, index_num=index['index_num'], index_cat=index['index_cat'], total_n_cat=9)
    print(x[:5, :])
    print(x.shape)
    token = tokenizer(x)
    print(token[:5, :, 0])
    print(token.shape)