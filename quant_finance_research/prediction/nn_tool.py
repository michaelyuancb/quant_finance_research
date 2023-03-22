import torch
from torch.utils.data import Dataset, DataLoader


def _transform_ndim_1(x):
    if x.ndim == 1:
        return x.reshape(-1, 1)
    else:
        return x


class TabDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(_transform_ndim_1(x)).float()
        self.y = torch.tensor(_transform_ndim_1(y)).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class TabLossDataset(Dataset):
    def __init__(self, x, y, help_loss):
        self.x = torch.tensor(_transform_ndim_1(x)).float()
        self.y = torch.tensor(_transform_ndim_1(y)).float()
        self.help_loss = torch.tensor(_transform_ndim_1(help_loss))

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.help_loss[index]

    def __len__(self):
        return self.x.shape[0]


def nn_train(dataloader, model, optimizer, loss_func, use_loss_column=False, device='gpu'):
    model.train()
    floss = torch.zeros(1).to(device)
    if use_loss_column:
        for (X, y, loss_help) in dataloader:
            X = X.to(device)
            y = y.to(device)
            loss_help = loss_help.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_func(pred, y, loss_help)
            loss.backward()
            optimizer.step()
            floss = floss + loss
    else:
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            floss = floss + loss

    return (floss.item()) / len(dataloader)


def nn_eval(dataloader, model, loss_func, use_loss_column=False, device='gpu'):
    model.eval()
    floss = torch.zeros(1).to(device)
    with torch.no_grad():
        if use_loss_column:
            for (X, y, loss_help) in dataloader:
                X, y = X.to(device), y.to(device)
                loss_help = loss_help.to(device)
                pred = model(X)
                loss = loss_func(pred, y, loss_help)
                floss = floss + loss
        else:
            for (X, y) in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_func(pred, y)
                floss = floss + loss

    return (floss.item()) / len(dataloader)


def nn_predict(dataloader, model, return_numpy=True, device='gpu'):
    pred_list = []
    model.eval()
    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_list.append(pred)
    pred = torch.concat(pred_list, dim=0)
    if return_numpy:
        pred = pred.numpy() if device == 'cpu' else pred.cpu().numpy()
    return pred
