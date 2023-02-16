import numpy as np

from quant_finance_research.eval.factor_eval import *


def generate_normal_pred():
    pred = np.array([0.1, 2.3, 5.4, 1.2, 1.79])
    label = np.array([0.15, 2.31, 5.48, 3.0, -2])
    time_id = [0, 0, 1, 1, 0]
    return pred, label, time_id


def generate_large_pred():
    pred = np.random.randn(1000)
    label = np.random.randn(2000)[1000:]
    time_id = []
    for i in range(20):
        for j in range(50):
            time_id.append(i)
    return pred, label, time_id


def generate_cord_pred():
    pred = np.array([1, 2, 3, 4, 5])
    time_id = [0, 0, 1, 1, 0]
    return pred, pred, time_id


def debug_evaluate():
    pred, label, _ = generate_normal_pred()
    print(f"evaluate_mse={evaluate_mse(pred, label)}")
    print(f"evaluate_rmse={evaluate_rmse(pred, label)}")
    print(f"evaluate_IC={evaluate_IC(pred, label)}")
    print(f"evaluate_RankIC={evaluate_RankIC(pred, label)}")
    print(f"evaluate_CCC={evaluate_CCC(pred, label)}")
    pred, label, _ = generate_cord_pred()
    print(f"evaluate_mse={evaluate_mse(pred, label)}")
    print(f"evaluate_rmse={evaluate_rmse(pred, label)}")
    print(f"evaluate_IC={evaluate_IC(pred, label)}")
    print(f"evaluate_RankIC={evaluate_RankIC(pred, label)}")
    print(f"evaluate_CCC={evaluate_CCC(pred, label)}")


def debug_evaluate_time():
    pred, label, time_id = generate_normal_pred()
    print(pred)
    print(label)
    print(time_id)
    print(f"evaluate_mse_time={evaluate_mse_time(pred, label, time_id)}")
    print(f"evaluate_rmse_time={evaluate_rmse_time(pred, label, time_id)}")
    print(f"evaluate_IC_time={evaluate_IC_time(pred, label, time_id)}")
    print(f"evaluate_RankIC_time={evaluate_RankIC_time(pred, label, time_id)}")
    print(f"evaluate_CCC_time={evaluate_CCC_time(pred, label, time_id)}")
    print(f"evaluate_IR_time={evaluate_IR_time(pred, label, time_id)}")
    print(f"evaluate_RankIR_time={evaluate_RankIR_time(pred, label, time_id)}")
    print(f"evaluate_classTop_acc_time_2c={evaluate_classTop_acc_time(pred, label, time_id, class_num=2)}")
    print(f"evaluate_classBottom_acc_time_2c={evaluate_classBottom_acc_time(pred, label, time_id, class_num=2)}")
    pred, label, time_id = generate_cord_pred()
    print(f"evaluate_mse_time={evaluate_mse_time(pred, label, time_id)}")
    print(f"evaluate_rmse_time={evaluate_rmse_time(pred, label, time_id)}")
    print(f"evaluate_IC_time={evaluate_IC_time(pred, label, time_id)}")
    print(f"evaluate_RankIC_time={evaluate_RankIC_time(pred, label, time_id)}")
    print(f"evaluate_CCC_time={evaluate_CCC_time(pred, label, time_id)}")
    print(f"evaluate_IR_time={evaluate_IR_time(pred, label, time_id)}")
    print(f"evaluate_RankIR_time={evaluate_RankIR_time(pred, label, time_id)}")
    print(f"evaluate_classTop_acc_time_2={evaluate_classTop_acc_time(pred, label, time_id, class_num=2)}")
    print(f"evaluate_classBottom_acc_time_2={evaluate_classBottom_acc_time(pred, label, time_id, class_num=2)}")
    pred, label, time_id = generate_large_pred()
    result_df = evaluate_factor_time_classic_1(pred, label, time_id, model_name="Debug")
    print(result_df)


if __name__ == "__main__":
    debug_evaluate_time()