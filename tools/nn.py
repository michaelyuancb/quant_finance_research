import copy
import random
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader


class TabDataset(Dataset):
    def __init__(self, x, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class NeuralNetworkWrapper:

    """Wrap the neural network framework for convenient usage."""
    """API
    
        solver = NeuralNetworkWrapper(dnn, device)
        
        solver.training(dataloader, loss_func, optimizer)  # Train one epoch, return loss on dataloader.
        
        solver.evaluate(dataloader, loss_func) # evaluate one epoch, return loss on dataloader.
        
        solver.train(self, xtrain, ytrain, xval, yval, 
                    loss_func, optimizer,
                    early_stop=20,       # Early stop is on validation dataset.
                    max_epoch=None,      # The max epoch number.
                    epoch_print=10,      
                    batch_size=2048):    # train the model with batch_size, no return.
                    
        solver.predict(xtest, batch_size=2048, transfer_cpu=True)  # predict the result. return the Tensor(N_test, **).
    """

    def __init__(self, dnn, device='cuda'):
        self.nn = dnn.to(device)
        self.best_param = {}
        self.train_loss = []
        self.val_loss = []
        self.best_loss = []
        self.device = device

    def training(self, dataloader, loss_func, optimizer):
        self.nn.train()
        floss = []
        for (X, y) in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            pred = self.nn(X)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            floss.append(loss.item())
        floss = np.mean(np.array(floss))
        return floss

    def evaluate(self, dataloader, loss_func):
        self.nn.eval()
        floss = []
        with torch.no_grad():
            for (X, y) in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.nn(X)
                loss = loss_func(pred, y)
                floss.append(loss.item())
        floss = np.mean(np.array(floss))
        return floss

    def train(self, xtrain, ytrain, xval, yval, loss_func, optimizer,
              early_stop=20,
              max_epoch=None,
              epoch_print=10,
              batch_size=2048):
        train_loader = DataLoader(TabDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TabDataset(xval, yval), batch_size=batch_size, shuffle=False)

        loss_train = self.evaluate(train_loader, loss_func)
        loss_val = self.evaluate(val_loader, loss_func)
        loss_val_best = loss_val
        print(f"[initial]: train_loss={np.round(loss_train, 5)} ; val_loss={np.round(loss_val, 5)} ; "
              f"val_loss_best={np.round(loss_val_best, 5)}")
        count = 0
        epoch = 0
        inf_num = 1000000000
        max_epoch = max_epoch if max_epoch is not None else inf_num
        for eph in range(max_epoch):
            loss_train = self.training(train_loader, loss_func, optimizer)
            loss_val = self.evaluate(val_loader, loss_func)
            if loss_val <= loss_val_best:
                loss_val_best = loss_val
                self.best_param = self.nn.state_dict()
                count = 0
            else:
                count = count + 1
            epoch = epoch + 1
            if epoch % epoch_print == 0:
                print(f"[Epoch {epoch}]: train_loss={np.round(loss_train, 5)} ; val_loss={np.round(loss_val, 5)} ; "
                      f"val_loss_best={np.round(loss_val_best, 5)}")
            self.train_loss.append(loss_train)
            self.val_loss.append(loss_val)
            self.best_loss.append(loss_val_best)
            if count > early_stop:
                print(f"Early Stop with Total Epoch={epoch}, Finish Training.")
                break
            if eph == inf_num - 1:
                print(f" top with Max Limitation Epoch={epoch}, Finish Training.")

    def predict(self, xtest, batch_size=2048, transfer_cpu=True):
        pred = []
        self.nn.eval()
        n_test = xtest.shape[0]
        with torch.no_grad():
            l = 0
            r = np.min([n_test, l + batch_size])
            while r != n_test:
                x = torch.tensor(xtest[l:r]).float().to(self.device)
                y = self.nn(x)
                pred.append(y)
                l = r
                r = np.min([n_test, l + batch_size])
            x = torch.tensor(xtest[l:r]).float().to(self.device)
            pred.append(self.nn(x))
        pred = torch.concat(pred, dim=0)
        if transfer_cpu:
            pred = pred.cpu().numpy()
        return pred


class NeuralNetworkAvgBaggingWrapper:
    """Wrap the average bagging neural network ensemble framework for convenient usage."""
    """API

        solver = NeuralNetworkAvgBaggingWrapper(dnn_list, optimizer_list, seed_list, device)
        
        solver.ensemble_set_seed(seed)  # set the global seed.

        solver.train(self, xtrain, ytrain, xval, yval, 
                    loss_func, optimizer,
                    early_stop=20,       # Early stop is on validation dataset.
                    max_epoch=None,      # The max epoch number.
                    epoch_print=10,      
                    batch_size=2048):    # train all model with batch_size, no return.

        solver.predict(xtest, batch_size=2048, transfer_cpu=True)  
                            # predict the result. 
                            # return Tensor(N_test, **). and List[the prediction of each model]
        solver.get_param()  # get the state_parameter of each model with dict("model1":..., "model2":..., ...)
        solver.load_param() # load the state_parameter of each model.
    """

    def __init__(self, dnn_list, optimzer_list, seed_list,  device="cuda"):
        self.seed_list = seed_list
        self.device = device
        self.n_dnn = len(seed_list)
        self.model_list = dnn_list
        self.optimizer_list = optimzer_list

    def ensemble_set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(self, xtrain, ytrain, xval, yval, loss_func, optimizer,
              early_stop=20,
              max_epoch=None,
              epoch_print=10,
              batch_size=2048):
        for i, seed in enumerate(self.seed_list):
            print(f"Start Traning Model{i}: seed={seed}")
            self.ensemble_set_seed(self.seed_list[i])
            self.model_list[i].train(xtrain, ytrain, xval, yval, loss_func, optimizer,
                                     early_stop=early_stop, max_epoch=max_epoch,
                                     epoch_print=epoch_print,
                                     batch_size=batch_size)

    def predict(self, xtest, transfer_cpu=True):
        predict_val = []
        for i in range(self.n_dnn):
            val = self.model_list[i].predict(xtest, transfer_cpu)
            predict_val.append(val)
        val = np.mean(np.array(predict_val), 0)
        return val, predict_val

    def get_param(self):
        param = {}
        for i in range(self.n_dnn):
            model_param = self.model_list[i].nn.to('cpu').state_dict()
            param['model' + str(i)] = model_param
        return param

    def load_param(self, param):
        for i in range(self.n_dnn):
            self.model_list[i].nn.to('cpu')
            self.model_list[i].nn.load_state_dict(param['model' + str(i)])
            self.model_list[i].nn.to(self.device)