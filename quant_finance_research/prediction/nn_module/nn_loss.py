from torch import nn
import pandas as pd
import numpy as np
import torch


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        return self.loss(x, y)


class StaticWeightMSELoss(nn.Module):

    def __init__(self):
        super(StaticWeightMSELoss, self).__init__()

    def forward(self, x, y, weight):
        loss = ((x - y) ** 2) * weight
        return torch.sum(loss)


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.loss(x, y))


class StaticWeightRMSELoss(nn.Module):

    def __init__(self):
        super(StaticWeightRMSELoss, self).__init__()
        self.loss = StaticWeightMSELoss()

    def forward(self, x, y, weight):
        return torch.sqrt(self.loss(x, y, weight))


class ICLoss(nn.Module):

    def __init__(self):
        super(ICLoss, self).__init__()
        pass

    def forward(self, x, y):
        x, y = x.reshape(-1), y.reshape(-1)
        x_mean, y_mean = x.mean(), y.mean()
        x_std, y_std = x.std(), y.std()
        x_z, y_z = (x - x_mean) / x_std, (y - y_mean) / y_std
        ic = torch.sum(x_z * y_z) / (x.shape[0] - 1)
        return -ic  # -pearson


class StaticWeightICLoss(nn.Module):

    def __init__(self):
        super(StaticWeightICLoss, self).__init__()
        pass

    def forward(self, x, y, weight):
        x, y = x.reshape(-1), y.reshape(-1)
        weight = weight.reshape(-1)
        x_mean, y_mean = (x * weight).mean(), (y * weight).mean()
        cov = torch.sum(weight * x * y) - x_mean * y_mean
        x_std = torch.sqrt(torch.sum(weight * (x ** 2)) - x_mean ** 2)
        y_std = torch.sqrt(torch.sum(weight * (y ** 2)) - y_mean ** 2)
        wic = cov / (x_std * y_std)
        return -wic  # -pearson


class CCCLoss(nn.Module):

    def __init__(self):
        super(CCCLoss, self).__init__()
        pass

    def forward(self, x, y):
        x, y = x.reshape(-1), y.reshape(-1)
        sxy = torch.sum((x - x.mean()) * (y - y.mean())) / (x.shape[0] - 1)
        ccc = 2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
        return -ccc  # -ccc


class StaticWeightCCCLoss(nn.Module):

    def __init__(self):
        super(StaticWeightCCCLoss, self).__init__()
        pass

    def forward(self, x, y, weight):
        x, y = x.reshape(-1), y.reshape(-1)
        weight = weight.reshape(-1)
        x_mean, y_mean = (x * weight).mean(), (y * weight).mean()
        cov = torch.sum(weight * x * y) - x_mean * y_mean
        x_var = torch.sum(weight * (x ** 2)) - x_mean ** 2
        y_var = torch.sum(weight * (y ** 2)) - y_mean ** 2
        ccc = 2 * cov / (x_var + y_var + (x_mean - y_mean) ** 2)
        return -ccc  # -ccc

