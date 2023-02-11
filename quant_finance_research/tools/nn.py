import abc
import itertools
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from quant_finance_research.tools.util import generate_cv_result_df, save_pickle, load_pickle


def dnn_set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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

        solver = NeuralNetworkWrapper(dnn, optimizer, device)

        solver.training(dataloader, loss_func)  # Train one epoch, return train_loss. [float]

        solver.evaluate(dataloader, loss_func) # evaluate one epoch, return eval_loss. [float]

        solver.train(self, xtrain, ytrain, xval, yval, loss_func,
                    early_stop=20,       # Early stop is on validation dataset.
                    max_epoch=None,      # The max epoch number.
                    epoch_print=10,     
                    num_workers=0, 
                    batch_size=2048):    # train the model with batch_size, return loss_val_best. [float]

        solver.predict(xtest, batch_size=2048)  
                # predict the result. return the np.array(N_test, **). [ndarray]
                
        solver.set_device(device)  # set the wrapper to the new device.
        
        solver.get_best_param()  # get the best training model's parameter, the state_dict is on 'cpu'.
        
        solver.load_param()  # load the parameter to the model.
        
        solver.load_best_param()  # load the parameter to the model.
    """

    def __init__(self, dnn, optimizer, device='cuda'):
        self.nn = dnn.to(device)
        self.optimizer = optimizer
        self.best_param = {}
        self.train_loss = []
        self.val_loss = []
        self.best_loss = []
        self.device = device

    def training(self, dataloader, loss_func):
        self.nn.train()
        floss = torch.zeros(1).to(self.device)
        for (X, y) in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.nn(X)
            loss = loss_func(pred, y)
            loss.backward()
            self.optimizer.step()
            floss = floss + loss
        return (floss.item()) / len(dataloader)

    def evaluate(self, dataloader, loss_func):
        self.nn.eval()
        floss = torch.zeros(1).to(self.device)
        with torch.no_grad():
            for (X, y) in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.nn(X)
                loss = loss_func(pred, y)
                floss = floss + loss
        return (floss.item()) / len(dataloader)

    def train(self, xtrain, ytrain, xval, yval, loss_func,
              seed=0,
              early_stop=20,
              max_epoch=None,
              epoch_print=10,
              num_workers=0,
              batch_size=2048):
        dnn_set_seed(seed)
        train_loader = DataLoader(TabDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
        val_loader = DataLoader(TabDataset(xval, yval), batch_size=batch_size, shuffle=False, num_workers=num_workers)

        loss_train = self.evaluate(train_loader, loss_func)
        loss_val = self.evaluate(val_loader, loss_func)
        self.best_param = self.nn.state_dict()
        loss_val_best = loss_val
        print(f"[initial]: train_loss={np.round(loss_train, 5)} ; val_loss={np.round(loss_val, 5)} ; "
              f"val_loss_best={np.round(loss_val_best, 5)}")
        count = 0
        epoch = 0
        inf_num = 1000000000
        max_epoch = max_epoch if max_epoch is not None else inf_num
        for eph in range(max_epoch):
            loss_train = self.training(train_loader, loss_func)
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
        self.nn.load_state_dict(self.best_param)
        if self.device != 'cpu':
            self.nn.to('cpu')
        self.best_param = self.nn.state_dict()
        if self.device != 'cpu':
            self.nn.to(self.device)
        return loss_val_best

    def predict(self, xtest, batch_size=2048):
        self.load_param(self.best_param)
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
        if self.device == 'cpu':
            pred = pred.numpy()
        else:
            pred = pred.cpu().numpy()
        return pred

    def set_device(self, device):
        self.device = device
        self.nn.to(self.device)

    def get_best_param(self):
        return self.best_param

    def load_param(self, param):
        self.nn.load_state_dict(param, strict=False)

    def load_best_param(self):
        self.load_param(self.best_param)

    def save_best_param_file(self, filename):
        save_pickle(filename, self.get_best_param())

    def load_param_file(self, filename):
        param = load_pickle(filename)
        self.load_param(param)


class NeuralNetworkEnsembleBase(abc.ABC):

    def __init__(self, dnn_list, optimzer_list, device='cuda'):
        self.device = device
        self.n_dnn = len(dnn_list)
        self.model_list = []
        for i in range(self.n_dnn):
            model = NeuralNetworkWrapper(dnn_list[i], optimzer_list[i], device)
            self.model_list.append(model)

    def predict(self, xtest, batch_size=2048):
        predict_val = []
        for i in tqdm(range(self.n_dnn)):
            val = self.model_list[i].predict(xtest, batch_size=batch_size)
            predict_val.append(val)
        predict_val = np.array(predict_val)
        val = np.mean(predict_val, 0)
        return val, predict_val  # float, ndarray(k,)

    def get_best_param(self):
        param = {}
        for i in range(self.n_dnn):
            model_param = self.model_list[i].get_best_param()
            param['model' + str(i)] = model_param
        return param  # param on cpu

    def load_param(self, param):
        for i in range(self.n_dnn):
            self.model_list[i].load_param(param['model' + str(i)])

    def save_best_param_file(self, filename):
        save_pickle(filename, self.get_best_param())

    def load_param_file(self, filename):
        param = load_pickle(filename)
        self.load_param(param)


class NeuralNetworkAvgBaggingEnsemble(NeuralNetworkEnsembleBase):
    """Wrap the average bagging neural network ensemble framework for convenient usage."""
    """API

        solver = NeuralNetworkAvgBaggingWrapper(dnn_list, optimizer_list, seed_list, device)

        solver.ensemble_set_seed(seed)  # set the global seed.

        solver.train(self, xtrain, ytrain, xval, yval, loss_func,
                    early_stop=20,       # Early stop is on validation dataset.
                    max_epoch=None,      # The max epoch number.
                    epoch_print=10,      
                    num_workers=0,
                    batch_size=2048):    # train all model with batch_size, 
                                         # return np.float, np.array(k,) [the loss_val_best list]

        solver.predict(xtest, batch_size=2048)  
                            # predict the result. 
                            # return return np.float, np.array(k,) ; mean prediction & [the prediction of each model]
        solver.get_best_param()  # get the best state_parameter with dict("model1":..., "model2":..., ...) [dict]
        solver.load_param() # load the state_parameter of each model.
    """

    def __init__(self, dnn_list, optimzer_list, seed_list, device="cuda"):
        super(NeuralNetworkAvgBaggingEnsemble, self).__init__(dnn_list, optimzer_list, device)
        self.seed_list = seed_list

    def train(self, xtrain, ytrain, xval, yval, loss_func,
              early_stop=20,
              max_epoch=None,
              epoch_print=10,
              num_workers=0,
              batch_size=2048):
        loss_val_best = []
        for i, seed in enumerate(self.seed_list):
            print(f"Start Traning Model{i}: seed={seed}")
            lval = self.model_list[i].train(xtrain, ytrain, xval, yval, loss_func,
                                            seed=self.seed_list[i],
                                            early_stop=early_stop, max_epoch=max_epoch,
                                            epoch_print=epoch_print,
                                            num_workers=num_workers,
                                            batch_size=batch_size)
            loss_val_best.append(lval)
        loss_val_best = np.array(loss_val_best)
        return np.mean(loss_val_best), loss_val_best


class NeuralNetworkCVEnsemble(NeuralNetworkEnsembleBase):
    """Wrap the cross validation neural network ensemble framework for convenient usage."""
    """API

        solver = NeuralNetworkAvgBaggingWrapper(dnn_list, optimizer_list, seed_list, device)

        solver.ensemble_set_seed(seed)  # set the global seed.

        solver.train(self, data_df, spliter, x_column, y_column, loss_func,
                    early_stop=20,       # Early stop is on validation dataset.
                    max_epoch=None,      # The max epoch number.
                    epoch_print=10,      
                    num_workers=0,
                    batch_size=2048):    # train all model with batch_size.
                                         # return np.float, np.array(k,) [the loss_val_best list]

        solver.predict(xtest, batch_size=2048)  
                            # predict the result. 
                            # return return np.float, np.array(k,) ; mean prediction & [the prediction of each model]
        solver.get_best_param()  # get the best state_parameter with dict("model1":..., "model2":..., ...) [dict]
        solver.load_param() # load the state_parameter of each model.
    """

    def __init__(self, dnn_list, optimzer_list, seed, device="cuda"):
        super(NeuralNetworkCVEnsemble, self).__init__(dnn_list, optimzer_list, device)
        self.seed = seed

    def train(self, data_df, spliter, x_column, y_column, loss_func,
              inv_col='investment_id',
              time_col='time_id',
              need_split=True,
              early_stop=20,
              max_epoch=None,
              epoch_print=10,
              num_workers=0,
              batch_size=2048):
        assert self.n_dnn == spliter.get_k()
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        loss_val_best = []
        for i in range(self.n_dnn):
            print(f"Start Traning Model{i}")
            xtr, ytr, xvl, yvl = spliter.get_folder(data_df, x_column, y_column, i)
            lval = self.model_list[i].train(xtr, ytr, xvl, yvl, loss_func,
                                            seed=self.seed,
                                            early_stop=early_stop, max_epoch=max_epoch,
                                            epoch_print=epoch_print,
                                            num_workers=num_workers,
                                            batch_size=batch_size)
            loss_val_best.append(lval)

        loss_val_best = np.array(loss_val_best)
        return np.mean(loss_val_best), loss_val_best


class NeuralNetworkCV:

    def __init__(self, dnn_list, optimzer_list, device="cuda"):
        self.device = device
        self.n_dnn = len(dnn_list)
        self.model_list = []
        for i in range(self.n_dnn):
            model = NeuralNetworkWrapper(dnn_list[i], optimzer_list[i], device)
            self.model_list.append(model)

    def cv(self, data_df, spliter, x_column, y_column, loss_func,
           inv_col='investment_id',
           time_col='time_id',
           need_split=True,
           early_stop=20,
           max_epoch=None,
           epoch_print=10,
           num_workers=0,
           batch_size=2048):
        assert self.n_dnn == spliter.get_k()
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        loss_val_list = []
        for i in range(self.n_dnn):
            print(f"Start Traning Model{i}")
            xtr, ytr, xvl, yvl = spliter.get_folder(data_df, x_column, y_column, i)
            loss_val = self.model_list[i].train(xtr, ytr, xvl, yvl, loss_func,
                                                seed=0,
                                                early_stop=early_stop, max_epoch=max_epoch,
                                                epoch_print=epoch_print,
                                                num_workers=num_workers,
                                                batch_size=batch_size)
            loss_val_list.append(loss_val)

        loss_val_list = np.array(loss_val_list)
        return np.mean(loss_val_list), loss_val_list

    def get_best_param(self):
        param = {}
        for i in range(self.n_dnn):
            model_param = self.model_list[i].get_best_param()
            param['model' + str(i)] = model_param
        return param  # param on cpu

    def load_param(self, param):
        for i in range(self.n_dnn):
            self.model_list[i].load_param(param['model' + str(i)])

    def save_best_param_file(self, filename):
        save_pickle(filename, self.get_best_param())

    def load_param_file(self, filename):
        param = load_pickle(filename)
        self.load_param(param)


class NeuralNetworkGridCVBase(abc.ABC):

    def __init__(self, k, param_dict, device='cuda'):
        self.param_dict = param_dict
        self.k = k
        self.device = device
        self.cv_param = []

    def cv(self, data_df, spliter, x_column, y_column, loss_func,
           inv_col='investment_id',
           time_col='time_id',
           need_split=True,
           early_stop=20,
           max_epoch=None,
           epoch_print=10,
           num_workers=0,
           batch_size=2048):
        assert spliter.get_k() == self.k
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        keys, values = zip(*self.param_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        cv_result = []
        for param in tqdm(permutations_dicts):
            dnn_list, optimizer_list = self.get_model_list(param)
            print(f"Start CV param={param}")
            cv_dnn = NeuralNetworkCV(dnn_list, optimizer_list, self.device)
            loss_mean, loss_val_list = cv_dnn.cv(data_df, spliter, x_column, y_column, loss_func,  # numpy
                                                 need_split=False,
                                                 early_stop=early_stop,
                                                 max_epoch=max_epoch,
                                                 epoch_print=epoch_print,
                                                 num_workers=num_workers,
                                                 batch_size=batch_size
                                                 )
            self.cv_param.append(cv_dnn.get_best_param())
            cv_result.append((loss_mean, loss_val_list))
        return generate_cv_result_df(cv_result, permutations_dicts)

    def get_cv_param(self, idx):
        """
        get the cross validation parameter (k model) of the idx hyper-combination.
        """
        return self.cv_param[idx]

    def get_cv_param_list(self):
        """
        get the cross validation parameter (k model) of the whole list (hyper-combination).
        """
        return self.cv_param

    def save_cv_param_list_file(self, filename):
        save_pickle(filename, self.get_cv_param_list())

    def get_model_list(self, param):
        """
        User should write their own function to get dnn_list & optimizer_list from param
        """
        dnn_list = []
        optimizer_list = []
        assert self.k == len(dnn_list)
        return dnn_list, optimizer_list


class NeuralNetworkGridCV_Example(NeuralNetworkGridCVBase):
    """
    An Example to show how to use NeuralNetworkGridCV
    """

    def __init__(self, k, param_dict, device='cuda'):
        super(NeuralNetworkGridCV_Example, self).__init__(k, param_dict, device)

    def get_model_list(self, param):
        """
        User should write their own function to get dnn_list & optimizer_list from param
        """
        from ..model.base_dnn import Base_DNN
        dnn_list = []
        optimizer_list = []
        param = {'learning_rate': 0.1}
        for i in range(self.k):
            dnn = Base_DNN(input_dim=5, hidden_dim=5, dropout_rate=0)
            optim = torch.optim.Adam(dnn.parameters(), lr=param['learning_rate'])
            dnn_list.append(dnn)
            optimizer_list.append(optim)
        assert self.k == len(dnn_list)
        return dnn_list, optimizer_list
