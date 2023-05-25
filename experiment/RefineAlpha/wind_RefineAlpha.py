import copy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import numpy as np
import itertools
import sys
import time
from quant_finance_research.prediction.nn_tool import TabDataset


class BaseMLP(nn.Module):

    def __init__(self, input_dim, num_hidden=2, hidden_dim=None, output_dim=None, activate=nn.ReLU(),
                 dropout=0.0, device='cuda'):
        super(BaseMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.num_hidden_dim = num_hidden
        self.output_dim = input_dim if output_dim is None else output_dim
        self.linear_hidden = nn.ModuleList([])
        self.linear_input = nn.Linear(self.input_dim, self.hidden_dim)
        for i in range(num_hidden):
            self.linear_hidden.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.linear_output = nn.Linear(self.hidden_dim, self.output_dim)
        self.activate = activate

    def forward(self, x):  # (B, K)
        x = self.dropout(x)
        x = self.activate(self.linear_input(x))
        for i in range(self.num_hidden_dim):
            x = self.activate(self.linear_hidden[i](x))
        x = self.linear_output(x)
        return x


class CasualMask(nn.Module):

    def __init__(self, input_dim, casual_ratio=0.8, num_hidden=2, hidden_dim=None, activate=nn.ReLU(),
                 mask_device='cuda'):
        super(CasualMask, self).__init__()
        self.input_dim = input_dim
        self.casual_ratio = casual_ratio
        self.casual_topk = int(input_dim * casual_ratio)
        self.mlp = BaseMLP(input_dim, hidden_dim=hidden_dim, num_hidden=num_hidden, output_dim=input_dim,
                           activate=activate)
        self.mask_device = mask_device
        self.environment_infer = BaseMLP(input_dim, hidden_dim=input_dim//2, num_hidden=num_hidden,
                                         output_dim=input_dim//4)

    def forward(self, tab_x):  # (N, K)
        tab_sft = self.mlp(tab_x)

        # # hard-mask
        # value, indices = torch.topk(tab_sft, self.casual_topk, dim=-1)
        # mask = torch.nn.functional.one_hot(indices, num_classes=self.input_dim).sum(dim=1).to(self.mask_device)
        # loss_mask = 0

        # soft-mask
        mask = torch.sigmoid(tab_sft)
        # loss_mask = torch.sum((mask.sum(dim=1) - self.casual_ratio) ** 2 + 10.0 *
        #                       torch.abs(mask.sum(dim=1) - self.casual_ratio))
        loss_mask = 0

        # for permutation invariant, the i_feat.shape[1] == v_feat.shape[1] == tab_x.shape[1]
        i_feat = tab_x * mask  # invariant feature.  (casual)
        v_feat = tab_x * (1.0 - mask)  # variant feature.    (environment)
        v_feat = self.environment_infer(v_feat)
        return i_feat, v_feat, loss_mask


def stable_tabnet_train(X, y, casual_masker, n_clusters, inv_predictor, lambda_var, lambda_mask,
                        optimizer, loss_func, batch_size, cluster_max_iter=300, device='cuda'):
    casual_masker.train()
    inv_predictor.train()
    dataloader_train = DataLoader(TabDataset(X, y), batch_size=batch_size, shuffle=True)
    floss = torch.zeros(1).to(device)
    floss_mean = torch.zeros(1).to(device)
    floss_var = torch.zeros(1).to(device)
    floss_mask = torch.zeros(1).to(device)

    for X, y in dataloader_train:
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        i_feat, v_feat, loss_mask = casual_masker(X)
        v_feat = v_feat.detach().numpy() if device == 'cpu' else v_feat.detach().cpu().numpy()

        cluster_clf = KMeans(n_clusters=n_clusters, n_init='auto', max_iter=cluster_max_iter, random_state=0)
        cluster_clf.fit(v_feat)
        cluster = cluster_clf.labels_

        cluster_loss = []
        for c in range(n_clusters):
            cluster_idx = np.where(cluster == c)[0]
            if cluster_idx.shape[0] == 0:
                continue
            pred_c = inv_predictor(i_feat[cluster_idx])
            loss = loss_func(pred_c, y[cluster_idx])
            cluster_loss.append(loss)

        loss_mean = torch.stack(cluster_loss, dim=0).mean(dim=0)
        loss_var = torch.stack(cluster_loss, dim=0).var(dim=0)
        loss = loss_mean + lambda_var * loss_var + lambda_mask * loss_mask
        floss = floss + loss
        floss_mean = floss_mean + loss_mean
        floss_var = floss_var + loss_var
        floss_mask = floss_mask + loss_mask
        loss.backward()
        optimizer.step()

    nd = len(dataloader_train)
    return floss.item() / nd, floss_mean.item() / nd, floss_var.item() / nd, floss_mask.item() / nd


def stable_tabnet_evaluate(X, y, casual_masker, n_clusters, inv_predictor, lambda_var, lambda_mask,
                           loss_func, batch_size,
                           cluster_max_iter=300, device='cuda'):
    casual_masker.eval()
    inv_predictor.eval()
    dataloader_val = DataLoader(TabDataset(X, y), batch_size=batch_size, shuffle=False)
    floss = torch.zeros(1).to(device)
    floss_mean = torch.zeros(1).to(device)
    floss_var = torch.zeros(1).to(device)
    floss_mask = torch.zeros(1).to(device)
    with torch.no_grad():
        for X, y in dataloader_val:
            X = X.to(device)
            y = y.to(device)
            i_feat, v_feat, loss_mask = casual_masker(X)
            v_feat = v_feat.numpy() if device == 'cpu' else v_feat.cpu().numpy()

            cluster_clf = KMeans(n_clusters=n_clusters, n_init='auto', max_iter=cluster_max_iter, random_state=0)
            cluster_clf.fit(v_feat)
            cluster = cluster_clf.labels_
            cluster_loss = []
            for c in range(n_clusters):
                cluster_idx = np.where(cluster == c)[0]
                if cluster_idx.shape[0] == 0:
                    continue
                pred_c = inv_predictor(i_feat[cluster_idx])
                loss = loss_func(pred_c, y[cluster_idx])
                cluster_loss.append(loss)

            loss_mean = torch.stack(cluster_loss, dim=0).mean(dim=0)
            loss_var = torch.stack(cluster_loss, dim=0).var(dim=0)
            loss = loss_mean + lambda_var * loss_var + lambda_mask * loss_mask
            floss = floss + loss
            floss_mean = floss_mean + loss_mean
            floss_var = floss_var + loss_var
            floss_mask = floss_mask + loss_mask

    nd = len(dataloader_val)
    return floss.item() / nd, floss_mean.item() / nd, floss_var.item() / nd, floss_mask.item() / nd


def stable_tabnet_predict(X, casual_masker, inv_predictor, batch_size, device='cuda'):
    casual_masker.eval()
    inv_predictor.eval()
    dataloader_test = DataLoader(TabDataset(X, X[:, 0]), batch_size=batch_size, shuffle=False)
    pred_result = []
    with torch.no_grad():
        for X, y in dataloader_test:
            X = X.to(device)
            i_feat, v_feat, loss_mask = casual_masker(X)
            pred = inv_predictor(i_feat)
            pred_result.append(pred)
    pred_result = torch.concat(pred_result, dim=0)
    pred_result = pred_result.numpy() if device == 'cpu' else pred_result.cpu().numpy()
    return pred_result


def verbose_print(epoch, loss_train, loss_mean_train, loss_var_train, loss_mask_train,
                  loss_val, loss_mean_val, loss_var_val, loss_mask_val, eph_start, best_val_loss):
    if epoch < 0:
        print(f"Init: ", end="")
    else:
        print(f"Epoch{epoch}: ", end="")
    print(f"loss_train={np.round(loss_train, 5)}, loss_mean_train={np.round(loss_mean_train, 5)}, loss_var_train="
          f"{np.round(loss_var_train, 5)}, loss_mask_train={np.round(loss_mask_train, 5)} ; ", end="")
    print(f"loss_val={np.round(loss_val, 5)}, loss_mean_val={np.round(loss_mean_val, 5)}, loss_var_val="
          f"{np.round(loss_var_val, 5)} ; loss_mask_val={np.round(loss_mask_val)} ; ", end="")
    print(f"best_val_loss={np.round(best_val_loss, 5)} ; epoch_time={time.time() - eph_start:.2f}s")


def stable_tabnet_solve(Xtrain, ytrain, Xval, yval, casual_masker, n_clusters, inv_predictor, lambda_var, lambda_mask,
                        optimizer, loss_func,
                        n_epochs,
                        early_stop_epochs,
                        batch_size,
                        verbose,
                        cluster_max_iter=300, device='cuda'):
    eph_start = time.time()
    loss_train, loss_mean_train, loss_var_train, loss_mask_train = \
        stable_tabnet_evaluate(Xtrain, ytrain, casual_masker=casual_masker, n_clusters=n_clusters,
                               inv_predictor=inv_predictor, lambda_var=lambda_var, lambda_mask=lambda_mask,
                               loss_func=loss_func,
                               batch_size=batch_size, cluster_max_iter=cluster_max_iter, device=device)
    loss_val, loss_mean_val, loss_var_val, loss_mask_val = \
        stable_tabnet_evaluate(Xval, yval, casual_masker=casual_masker, n_clusters=n_clusters,
                               inv_predictor=inv_predictor, lambda_var=lambda_var, lambda_mask=lambda_mask,
                               loss_func=loss_func,
                               batch_size=batch_size, cluster_max_iter=cluster_max_iter, device=device)
    loss_val_best = loss_val
    verbose_print(-1, loss_train, loss_mean_train, loss_var_train, loss_mask_train,
                  loss_val, loss_mean_val, loss_var_val, loss_mask_val, eph_start, loss_val)
    early_stop_count = 0
    best_param_casual_masker = casual_masker.state_dict()
    best_param_inv_predictor = inv_predictor.state_dict()
    for epoch in range(n_epochs):
        eph_start = time.time()
        cc = copy.deepcopy(casual_masker)
        loss_train, loss_mean_train, loss_var_train, loss_mask_train = \
            stable_tabnet_train(Xtrain, ytrain, casual_masker=casual_masker, n_clusters=n_clusters,
                                inv_predictor=inv_predictor, lambda_var=lambda_var, lambda_mask=lambda_mask,
                                optimizer=optimizer, loss_func=loss_func,
                                batch_size=batch_size, cluster_max_iter=cluster_max_iter, device=device)
        # print(cc.parameters().__dict__)
        # print(casual_masker.parameters())
        # assert cc.parameters() == casual_masker.parameters()
        loss_val, loss_mean_val, loss_var_val, loss_mask_val = \
            stable_tabnet_evaluate(Xval, yval, casual_masker=casual_masker, n_clusters=n_clusters,
                                   inv_predictor=inv_predictor, lambda_var=lambda_var, lambda_mask=lambda_mask,
                                   loss_func=loss_func,
                                   batch_size=batch_size, cluster_max_iter=cluster_max_iter, device=device)
        if loss_val < loss_val_best:
            loss_val_best = loss_val
            best_param_casual_masker = casual_masker.state_dict()
            best_param_inv_predictor = inv_predictor.state_dict()
            early_stop_count = 0
        else:
            early_stop_count = early_stop_count + 1
        if verbose > 0 and epoch % verbose == 0:
            verbose_print(epoch, loss_train, loss_mean_train, loss_var_train, loss_mask_train,
                          loss_val, loss_mean_val, loss_var_val, loss_mask_val, eph_start, loss_val)
        if early_stop_count == early_stop_epochs:
            print(f"Early Stop in Epoch{epoch} ; ")
            break
    if early_stop_count != early_stop_epochs:
        print(f"Finish Training, total epoch={n_epochs}")
    casual_masker.load_state_dict(best_param_casual_masker)
    inv_predictor.load_state_dict(best_param_inv_predictor)


if __name__ == "__main__":
    input_dim = 12
    batch_size = 32
    X = np.random.randn(batch_size, input_dim)
    y = np.random.randn(batch_size, 1)
    casual_masker = CasualMask(input_dim, casual_ratio=0.6, activate=nn.ReLU()).to('cuda')
    inv_predictor = BaseMLP(input_dim, num_hidden=2, output_dim=1).to('cuda')
    optimizer = torch.optim.Adam(itertools.chain(casual_masker.parameters(), inv_predictor.parameters()), lr=1e-2)

    # loss, loss_mean, loss_var = stable_tabnet_train(X, y, casual_masker=casual_masker, n_clusters=4,
    #                                                 inv_predictor=inv_predictor, lambda_var=1e-2, lambda_mask=1e-1,
    #                                                 optimizer=optimizer,
    #                                                 loss_func=nn.MSELoss(), batch_size=8,
    #                                                 device='cuda')
    # print(f"loss={loss}")
    # print(f"loss_mean={loss_mean}")
    # print(f"loss_var={loss_var}")
    # loss, loss_mean, loss_var = stable_tabnet_evaluate(X, y, casual_masker=casual_masker, n_clusters=4,
    #                                                    inv_predictor=inv_predictor, lambda_var=1e-2, lambda_mask=1e-1,
    #                                                    loss_func=nn.MSELoss(), batch_size=8,
    #                                                    device='cuda')
    # print(f"loss={loss}")
    # print(f"loss_mean={loss_mean}")
    # print(f"loss_var={loss_var}")

    stable_tabnet_solve(X, y, X, y, casual_masker=casual_masker, n_clusters=4, inv_predictor=inv_predictor,
                        lambda_var=0.1, lambda_mask=0.5, optimizer=optimizer, loss_func=nn.MSELoss(),
                        n_epochs=1000, early_stop_epochs=20,
                        batch_size=8,
                        verbose=1, cluster_max_iter=300, device='cuda')

    pred = stable_tabnet_predict(X, casual_masker=casual_masker, inv_predictor=inv_predictor,
                                 batch_size=4, device='cuda')
    print(pred)
    print(pred.shape)
    print("success.")
