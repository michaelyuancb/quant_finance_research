
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch.cuda

from quant_finance_research.tools.nn import *
from quant_finance_research.tools.tscv_df import QuantTimeSplit_PurgeGroupTime, QuantTimeSplit_PurgeSeqPast
from quant_finance_research.model.base_dnn import Base_DNN
from quant_finance_research.tools.util import *
from quant_finance_research.tools.factor_eval import *




def get_data():
    _, train, xtrain, ytrain, xval, yval, xtest_lb, ytest_lb, \
    xtest_fn, ytest_fn, time_id_lb, time_id_fn = \
        load_pickle('result/split.pkl')
    return train, xtrain, ytrain, xval, yval, xtest_lb, ytest_lb, \
           xtest_fn, ytest_fn, time_id_lb, time_id_fn


train, xtrain, ytrain, xval, yval, xtest_lb, ytest_lb, \
    xtest_fn, ytest_fn, time_id_lb, time_id_fn = get_data()
print("Finish Loading Data.")
input_dim = xtrain.shape[1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_column = train.shape[1]
x_column = [i for i in range(2, n_column - 1)]
y_column = [n_column - 1]
batch_size = 4096
num_workers = 4


def solve_loss_function(task_id, model_name, k_fold=5):
    if task_id == 0:
        loss_func = MSELoss()
    elif task_id == 1:
        loss_func = ICLoss()
    elif task_id == 2:
        loss_func = CCCLoss()
    elif task_id == 3 or task_id == 4:
        loss_func = MSELoss()
    else:
        loss_func = MSELoss()
    dnn_solver = None

    lr = 1e-4

    if task_id <=2:
        dnn = Base_DNN(input_dim=input_dim, hidden_dim=int(1.5*input_dim), dropout_rate=0)
        optimizer = torch.optim.Adam(dnn.parameters(), lr=lr)
        dnn_solver = NeuralNetworkWrapper(dnn, optimizer, device=device)
        dnn_solver.train(xtrain, ytrain, xval, yval, loss_func,
                        early_stop=20,       # Early stop is on validation dataset.
                        max_epoch=None,      # The max epoch number.
                        epoch_print=10,
                        num_workers=num_workers,
                        batch_size=batch_size)
    elif task_id == 6:
        dnn = Base_DNN(input_dim=input_dim, hidden_dim=int(1.0*input_dim), dropout_rate=0)
        optimizer = torch.optim.Adam(dnn.parameters(), lr=0.0001)
        dnn_solver = NeuralNetworkWrapper(dnn, optimizer, device=device)
        dnn_solver.train(xtrain, ytrain, xval, yval, loss_func,
                        early_stop=20,       # Early stop is on validation dataset.
                        max_epoch=None,      # The max epoch number.
                        epoch_print=10,
                        num_workers=num_workers,
                        batch_size=batch_size)
    elif task_id == 7:
        dnn = Base_DNN(input_dim=input_dim, hidden_dim=int(1.0*input_dim), dropout_rate=0)
        optimizer = torch.optim.Adam(dnn.parameters(), lr=0.001)
        dnn_solver = NeuralNetworkWrapper(dnn, optimizer, device=device)
        dnn_solver.train(xtrain, ytrain, xval, yval, loss_func,
                        early_stop=20,       # Early stop is on validation dataset.
                        max_epoch=None,      # The max epoch number.
                        epoch_print=10,
                        num_workers=num_workers,
                        batch_size=batch_size)
    elif 3 <= task_id <= 4:
        dnn_list = []
        optim_list = []
        for i in range(k_fold):
            dnn = Base_DNN(input_dim=input_dim, hidden_dim=int(1.5*input_dim), dropout_rate=0)
            optimizer = torch.optim.Adam(dnn.parameters(), lr=lr)
            dnn_list.append(dnn)
            optim_list.append(optimizer)
        if task_id == 3:
            dnn_solver = NeuralNetworkAvgBaggingEnsemble(dnn_list=dnn_list, optimzer_list=optim_list,
                                                         seed_list=[i for i in range(k_fold)], device=device)
            dnn_solver.train(xtrain, ytrain, xval, yval, loss_func,
                            early_stop=20,       # Early stop is on validation dataset.
                            max_epoch=None,      # The max epoch number.
                            epoch_print=10,
                            num_workers=num_workers,
                            batch_size=batch_size)
        elif task_id == 4:
            dnn_solver = NeuralNetworkCVEnsemble(dnn_list=dnn_list, optimzer_list=optim_list,
                                                 seed=0, device=device)
            fsplit = QuantTimeSplit_PurgeGroupTime(k=k_fold, gap=1)
            dnn_solver.train(train, fsplit, x_column, y_column, loss_func,
                             need_split=True,
                             early_stop=20,
                             max_epoch=None,
                             epoch_print=10,
                             num_workers=num_workers,
                             batch_size=batch_size)
    param = dnn_solver.get_best_param()
    pred_lb = dnn_solver.predict(xtest_lb, batch_size=batch_size)
    pred_fn = dnn_solver.predict(xtest_fn, batch_size=batch_size)
    if task_id <= 2 or 6 <= task_id <= 7:
        pred_lb, pred_fn = pred_lb.reshape(-1), pred_fn.reshape(-1)
        eval_lb = evaluate_factor(pred_lb, ytest_lb, time_id_lb, model_name=model_name)
        eval_fn = evaluate_factor(pred_fn, ytest_fn, time_id_fn, model_name=model_name)
    else:
        pred_lb = (pred_lb[0].reshape(-1), pred_lb[1])
        pred_fn = (pred_fn[0].reshape(-1), pred_fn[1])
        eval_lb = evaluate_factor(pred_lb[0], ytest_lb, time_id_lb, model_name=model_name)
        eval_fn = evaluate_factor(pred_fn[0], ytest_fn, time_id_fn, model_name=model_name)
    
    print(eval_lb)
    if task_id <= 2 or 6 <= task_id <= 7:
        save_pickle('result/model_performance/'+model_name+'.pkl',
                    (param, pred_lb, pred_fn, eval_lb, eval_fn))
    else:
        save_pickle('result/model_performance/'+model_name+'.pkl',
                    (param, pred_lb[0], pred_fn[0], eval_lb, eval_fn, pred_lb[1], pred_fn[1]))

class NeuralNetworkGridCV(NeuralNetworkGridCVBase):
    """
    An Example to show how to use NeuralNetworkGridCV
    """
    def __init__(self, k, param_dict, device='cuda'):
        super(NeuralNetworkGridCV, self).__init__(k, param_dict, device)

    def get_model_list(self, param):
        """
        User should write their own function to get dnn_list & optimizer_list from param
        """
        dnn_list = []
        optimizer_list = []
        for i in range(self.k):
            dnn = Base_DNN(input_dim=input_dim, hidden_dim=int(param['dilate']*input_dim), dropout_rate=0)
            optim = torch.optim.Adam(dnn.parameters(), lr=param['learning_rate'])
            dnn_list.append(dnn)
            optimizer_list.append(optim)
        assert self.k == len(dnn_list)
        return dnn_list, optimizer_list


def solve_cv():
    param_dict = {'dilate':[1.0, 1.5, 2.0], 'learning_rate':[1e-3, 1e-4, 1e-5]}
    fsplit = QuantTimeSplit_PurgeSeqPast(k=5, gap=1)
    k = 5
    grid_cv = NeuralNetworkGridCV(k=5, param_dict=param_dict, device=device)
    loss_func = nn.MSELoss()
    df = grid_cv.cv(train, fsplit, x_column, y_column, loss_func,
                    need_split=True,
                    early_stop=20,
                    max_epoch=None,
                    epoch_print=10,
                    num_workers=num_workers,
                    batch_size=batch_size)
    df.to_csv('result/nn_cv_result.csv')
    

def solve_cv_fe():
    print("solve_cv_fe")
    
    df_fe = pd.read_csv('result/fe_train_df.csv', index_col=0)
    k_fold = 5
    fsplit = QuantTimeSplit_PurgeSeqPast(k=5, gap=1)
    loss_func = nn.MSELoss()

    xx_column = x_column + [i for i in range(303, df_fe.shape[1])]
    yy_column = y_column
    
    print(f"shape={df_fe.shape}")
    print(f"xx={len(xx_column)}")
    dnn_list = []
    optimizer_list = []
    for i in range(k_fold):
        dnn = Base_DNN(input_dim=len(xx_column), hidden_dim=int(1.5*len(xx_column)), dropout_rate=0)
        optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-4)
        dnn_list.append(dnn)
        optimizer_list.append(optimizer)
        
    cver = NeuralNetworkCV(dnn_list, optimizer_list, device="cuda")
    
    cv_mean_fe, cv_list_fe = cver.cv(df_fe, fsplit, xx_column, yy_column, loss_func, num_workers=num_workers, batch_size=batch_size)
    
    
    dnn_list = []
    optimizer_list = []
    for i in range(k_fold):
        dnn = Base_DNN(input_dim=input_dim, hidden_dim=int(1.5*input_dim), dropout_rate=0)
        optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-4)
        dnn_list.append(dnn)
        optimizer_list.append(optimizer)
        
    cver = NeuralNetworkCV(dnn_list, optimizer_list, device="cuda")
        
    cv_mean, cv_list = cver.cv(train, fsplit, x_column, y_column, loss_func, num_workers=num_workers, batch_size=batch_size)

    save_pickle('result/nn_fe_cv_result.pkl', ("cv_list, cv_list_fe", cv_list, cv_list_fe))

    print(f"cv_mean={np.round(cv_mean,5)}; cv_std={np.round(np.std(np.array(cv_list)),5)} ; cv_list={cv_list}")
    print(f"cv_mean_fe={np.round(cv_mean_fe,5)}; cv_std={np.round(np.std(np.array(cv_list_fe)),5)} ;  cv_list_fe={cv_list_fe} ")
    print(type(cv_list))
    
    

def solve_nn_fe(task_id, model_name):
    print(model_name)
    
    _, fe_train, fe_xtrain, fe_xval = load_pickle('result/fe_array_train.pkl')
    _, fe_xtest_lb, fe_xtest_fn = load_pickle('result/fe_array_test.pkl')
    loss_func = nn.MSELoss()
    input_dim = fe_train.shape[1]
    
    lr = 1e-4
    
    if task_id == 9:
        dnn = Base_DNN(input_dim=input_dim, hidden_dim=int(1.5*input_dim), dropout_rate=0)
    
    print(input_dim)
        
    optimizer = torch.optim.Adam(dnn.parameters(), lr=lr)
    dnn_solver = NeuralNetworkWrapper(dnn, optimizer, device=device)
    dnn_solver.train(fe_xtrain, ytrain, fe_xval, yval, loss_func,
                     early_stop=20,       # Early stop is on validation dataset.
                     max_epoch=None,      # The max epoch number.
                     epoch_print=10,
                     num_workers=num_workers,
                     batch_size=batch_size)
                        
    param = dnn_solver.get_best_param()
    pred_lb = dnn_solver.predict(fe_xtest_lb, batch_size=batch_size)
    pred_fn = dnn_solver.predict(fe_xtest_fn, batch_size=batch_size)
    pred_lb, pred_fn = pred_lb.reshape(-1), pred_fn.reshape(-1)
    eval_lb = evaluate_factor(pred_lb, ytest_lb, time_id_lb, model_name=model_name)
    eval_fn = evaluate_factor(pred_fn, ytest_fn, time_id_fn, model_name=model_name)
    save_pickle('result/model_performance/'+model_name+'.pkl', (param, pred_lb, pred_fn, eval_lb, eval_fn))


if __name__ == "__main__":
    task_name = {
        0: "nn_mse_baseline",
        1: "nn_ic",
        2: "nn_ccc",
        3: "nn_bg_ensemble",
        4: "nn_cv_ensemble",
        5: "nn_cv_find",
        6: "nn_cv_mean",
        7: "nn_cv_meanstd",
        8: "nn_cv_fe",
        9: "nn_fe_mse"
    }
    task_id = 9

    if task_id in [0, 1, 2, 3, 4, 6, 7]:
        solve_loss_function(task_id, task_name[task_id], k_fold=5)
    elif task_id == 8:
        solve_cv_fe()
    elif task_id == 9:
        solve_nn_fe(task_id, task_name[task_id])
    elif task_id == 5:
        solve_cv()
