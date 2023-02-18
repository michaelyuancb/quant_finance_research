import numpy as np

from quant_finance_research.eval.factor_eval import *
from datetime import datetime


def generate_normal_pred():
    pred = np.array([0.1, 2.3, 5.4, 7.6, 10.0])
    label = np.array([0.15, 2.0, 6.4, 5.4, 11.2])
    t1 = datetime.strptime("2022-11-1", "%Y-%m-%d")
    t2 = datetime.strptime("2022-11-2", "%Y-%m-%d")
    time_id = np.array([t1, t1, t2, t2, t1])
    # time_id = np.array([0, 0, 1, 1, 0])
    weight = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
    return pred, label, time_id, weight


def generate_large_pred():
    pred = np.random.randn(1000)
    label = np.random.randn(2000)[1000:]
    time_id = []
    for i in range(20):
        for j in range(50):
            time_id.append(i)
    time_id = np.array(time_id)
    weight = np.random.rand(1000)
    weight = weight / np.sum(weight)
    return pred, label, time_id, weight


def generate_cord_pred():
    pred = np.array([1, 2, 3, 4, 5])
    t1 = datetime.strptime("2022-11-1", "%Y-%m-%d")
    t2 = datetime.strptime("2022-11-2", "%Y-%m-%d")
    time_id = np.array([t1, t1, t2, t2, t1])
    weight = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
    return pred, pred, time_id, weight


def debug_evaluate():
    print("========Rand=========")
    pred, label, _, _ = generate_normal_pred()
    print(f"evaluate_MSE={evaluate_MSE(pred, label)}")
    print(f"evaluate_RMSE={evaluate_RMSE(pred, label)}")
    print(f"evaluate_IC={evaluate_IC(pred, label)}")
    print(f"evaluate_RankIC={evaluate_RankIC(pred, label)}")
    print(f"evaluate_CCC={evaluate_CCC(pred, label)}")
    print(f"evaluate_classTop_acc_time={evaluate_classTop_acc(pred, label, class_num=2)}")
    print(f"evaluate_classBottom_acc={evaluate_classBottom_acc(pred, label, class_num=2)}")

    print("========Cord=========")
    pred, label, _, _ = generate_cord_pred()
    print(f"evaluate_MSE={evaluate_MSE(pred, label)}")
    print(f"evaluate_RMSE={evaluate_RMSE(pred, label)}")
    print(f"evaluate_IC={evaluate_IC(pred, label)}")
    print(f"evaluate_RankIC={evaluate_RankIC(pred, label)}")
    print(f"evaluate_CCC={evaluate_CCC(pred, label)}")
    print(f"evaluate_classTop_acc_time={evaluate_classTop_acc(pred, label, class_num=2)}")
    print(f"evaluate_classBottom_acc={evaluate_classBottom_acc(pred, label, class_num=2)}")

    print("=========Large=========")
    pred, label, time_id, _ = generate_large_pred()
    result_df = evaluate_factor_classic_1(pred, label, model_name="Debug")
    print(result_df)


def debug_evaluate_time():
    print("========Rand=========")
    pred, label, time_id, _ = generate_normal_pred()
    print(f"evaluate_MSE_time={evaluate_MSE_time(pred, label, time_id)}")
    print(f"evaluate_RMSE_time={evaluate_RMSE_time(pred, label, time_id)}")
    print(f"evaluate_IC_time={evaluate_IC_time(pred, label, time_id)}")
    print(f"evaluate_RankIC_time={evaluate_RankIC_time(pred, label, time_id)}")
    print(f"evaluate_CCC_time={evaluate_CCC_time(pred, label, time_id)}")
    print(f"evaluate_IR_time={evaluate_IR_time(pred, label, time_id)}")
    print(f"evaluate_RankIR_time={evaluate_RankIR_time(pred, label, time_id)}")
    print(f"evaluate_classTop_acc_time_2c={evaluate_classTop_acc_time(pred, label, time_id, class_num=2)}")
    print(pred)
    print(label)
    print(f"evaluate_classBottom_acc_time_2c={evaluate_classBottom_acc_time(pred, label, time_id, class_num=2)}")

    print("========Cord=========")
    pred, label, time_id, _ = generate_cord_pred()
    print(f"evaluate_MSE_time={evaluate_MSE_time(pred, label, time_id)}")
    print(f"evaluate_RMSE_time={evaluate_RMSE_time(pred, label, time_id)}")
    print(f"evaluate_IC_time={evaluate_IC_time(pred, label, time_id)}")
    print(f"evaluate_RankIC_time={evaluate_RankIC_time(pred, label, time_id)}")
    print(f"evaluate_CCC_time={evaluate_CCC_time(pred, label, time_id)}")
    print(f"evaluate_IR_time={evaluate_IR_time(pred, label, time_id)}")
    print(f"evaluate_RankIR_time={evaluate_RankIR_time(pred, label, time_id)}")
    print(f"evaluate_classTop_acc_time_2={evaluate_classTop_acc_time(pred, label, time_id, class_num=2)}")
    print(f"evaluate_classBottom_acc_time_2={evaluate_classBottom_acc_time(pred, label, time_id, class_num=2)}")

    print("=========Large=========")
    pred, label, time_id, _ = generate_large_pred()
    result_df = evaluate_factor_time_classic_1(pred, label, time_id, model_name="Debug")
    print(result_df)


def debug_evaluate_StaticWeight():
    print("========Rand=========")
    pred, label, time_id, weight = generate_normal_pred()
    print(f"evaluate_MSE_StaticWeight={evaluate_MSE_StaticWeight(pred, label, weight)}")
    print(f"evaluate_RMSE_StaticWeight={evaluate_RMSE_StaticWeight(pred, label, weight)}")
    print(f"evaluate_IC_StaticWeight={evaluate_IC_StaticWeight(pred, label, weight)}")
    print(f"evaluate_RankIC_StaticWeight={evaluate_RankIC_StaticWeight(pred, label, weight)}")
    print(f"evaluate_CCC_StaticWeight={evaluate_CCC_StaticWeight(pred, label, weight)}")

    print("========Cord=========")
    pred, label, time_id, weight = generate_cord_pred()
    print(f"evaluate_MSE_StaticWeight={evaluate_MSE_StaticWeight(pred, label, weight)}")
    print(f"evaluate_RMSE_StaticWeight={evaluate_RMSE_StaticWeight(pred, label, weight)}")
    print(f"evaluate_IC_StaticWeight={evaluate_IC_StaticWeight(pred, label, weight)}")
    print(f"evaluate_RankIC_StaticWeight={evaluate_RankIC_StaticWeight(pred, label, weight)}")
    print(f"evaluate_CCC_StaticWeight={evaluate_CCC_StaticWeight(pred, label, weight)}")

    print("=========Large=========")
    pred, label, time_id, weight = generate_large_pred()
    result_df = evaluate_factor_StaticWeight_classic_1(pred, label, weight, model_name="Debug")
    print(result_df)


def debug_evaluate_StaticWeight_time():
    print("========Rand=========")
    pred, label, time_id, weight = generate_normal_pred()
    print(f"evaluate_MSE_StaticWeight_time={evaluate_MSE_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_RMSE_StaticWeight_time={evaluate_RMSE_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_IC_StaticWeight_time={evaluate_IC_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_RankIC_StaticWeight_time={evaluate_RankIC_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_CCC_StaticWeight_time={evaluate_CCC_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_IR_StaticWeight_time={evaluate_IR_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_RankIR_StaticWeight_time={evaluate_RankIR_StaticWeight_time(pred, label, weight, time_id)}")

    print("========Cord=========")
    pred, label, time_id, weight = generate_cord_pred()
    print(f"evaluate_MSE_StaticWeight_time={evaluate_MSE_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_RMSE_StaticWeight_time={evaluate_RMSE_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_IC_StaticWeight_time={evaluate_IC_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_RankIC_StaticWeight_time={evaluate_RankIC_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_CCC_StaticWeight_time={evaluate_CCC_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_IR_StaticWeight_time={evaluate_IR_StaticWeight_time(pred, label, weight, time_id)}")
    print(f"evaluate_RankIR_StaticWeight_time={evaluate_RankIR_StaticWeight_time(pred, label, weight, time_id)}")

    print("=========Large=========")
    pred, label, time_id, weight = generate_large_pred()
    result_df = evaluate_factor_StaticWeight_time_classic_1(pred, label, weight, time_id, model_name="Debug")
    print(result_df)


def debug_evaluate_ic():

    print("========Rand=========")
    pred, label, time_id, _ = generate_normal_pred()
    print(f"evaluate_IC_time={evaluate_IC_time(pred, label, time_id)}")


if __name__ == "__main__":
    # debug_evaluate()
    # debug_evaluate_time()
    # debug_evaluate_StaticWeight()
    debug_evaluate_StaticWeight_time()
