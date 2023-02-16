import torch

from quant_finance_research.eval.nn_loss import *
from quant_finance_research.eval.factor_eval import *
import numpy as np
import pandas as pd


class DebugICLoss:

    def debug_forward(self):
        x = np.array([1., 2., 3., 4.])
        y = np.array([1., 3., 4., 19.])
        print(evaluate_IC(x, y))
        loss = ICLoss()
        print(loss(torch.tensor(x), torch.tensor(y)))


class DebugStaticWeightICLoss:

    def debug_forward(self):
        x = np.array([1., 2., 3., 4.])
        y = np.array([1., 2., 3., 4.])
        weight = np.array([0.1, 0.3, 0.4, 0.2])
        loss = StaticWeightICLoss()
        print(loss(torch.tensor(x), torch.tensor(y), torch.tensor(weight)))


class DebugCCCLoss:

    def debug_forward(self):
        x = np.array([1., 2., 3., 4.])
        y = np.array([1., 2., 3., 4.])
        weight = np.array([0.1, 0.3, 0.4, 0.2])
        loss = CCCLoss()
        print(loss(torch.tensor(x), torch.tensor(y)))


class DebugStaticWeightCCCLoss:

    def debug_forward(self):
        x = np.array([1., 2., 3., 4.])
        y = np.array([1., 2., 3., 4.])
        weight = np.array([0.1, 0.3, 0.4, 0.2])
        loss = StaticWeightCCCLoss()
        print(loss(torch.tensor(x), torch.tensor(y), torch.tensor(weight)))


if __name__ == "__main__":
    DebugICLoss().debug_forward()
    DebugStaticWeightICLoss().debug_forward()
    DebugCCCLoss().debug_forward()
    DebugStaticWeightCCCLoss().debug_forward()