import copy

from quant_finance_research.fe.fe_orth import *
from quant_finance_research.utils import *


def _get_fX_df(rest=5):
    df, df_column = get_example_large_df()
    for i in range(rest, 200):
        del df['factor_' + str(i)]
    df_column['x'] = [i for i in range(2, 2 + rest)]
    df_column['y'] = [2 + rest + 1]
    df_column['loss'] = [2 + rest + 2]
    return df, df_column


def debug_orth_panel_global_symmetry():
    df, df_column = _get_fX_df(rest=8)
    df_val = copy.deepcopy(df)
    top_k = None
    df, pro_pack = orth_panel_global_symmetry(df, df_column, column=df_column['x'], top_k=top_k)
    factor_data = df.iloc[:, df_column['x']].values
    print(f"x0**2 = {np.sum(factor_data[:, 0] * factor_data[:, 0])}")
    print(f"x1**2 = {np.sum(factor_data[:, 1] * factor_data[:, 1])}")
    print(f"x0*x1 = {np.sum(factor_data[:, 0] * factor_data[:, 1])}")
    df_val, _ = orth_panel_global_symmetry(df_val, df_column, column=df_column['x'], top_k=top_k, **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_panel_global_standard():
    df, df_column = _get_fX_df(rest=8)
    df_val = copy.deepcopy(df)
    top_k = None
    df, pro_pack = orth_panel_global_standard(df, df_column, column=df_column['x'], top_k=top_k)
    factor_data = df.iloc[:, df_column['x']].values
    print(f"x0**2 = {np.sum(factor_data[:, 0] * factor_data[:, 0])}")
    print(f"x1**2 = {np.sum(factor_data[:, 1] * factor_data[:, 1])}")
    print(f"x0*x1 = {np.sum(factor_data[:, 0] * factor_data[:, 1])}")
    df_val, _ = orth_panel_global_symmetry(df_val, df_column, column=df_column['x'], top_k=top_k, **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_panel_global_custom():
    df, df_column = _get_fX_df(rest=4)
    df_val = copy.deepcopy(df)
    top_k = None
    A = np.random.rand(4, 4)
    U, S, V = np.linalg.svd(A)
    custom = np.dot(U, V)
    print(custom)
    print(np.dot(custom.T, custom))
    df, pro_pack = orth_panel_global_custom(df, df_column, column=df_column['x'], custom_transform=custom, top_k=top_k)
    factor_data = df.iloc[:, df_column['x']].values
    print(f"x0**2 = {np.sum(factor_data[:, 0] * factor_data[:, 0])}")
    print(f"x1**2 = {np.sum(factor_data[:, 1] * factor_data[:, 1])}")
    print(f"x0*x1 = {np.sum(factor_data[:, 0] * factor_data[:, 1])}")
    df_val, _ = orth_panel_global_symmetry(df_val, df_column, column=df_column['x'], top_k=top_k, **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_panel_global_residual():
    df, df_column = _get_fX_df(rest=3)
    print(df)
    df_val = copy.deepcopy(df)
    top_k = None
    df, pro_pack = orth_panel_global_residual(df, df_column, column=df_column['x'])
    print(df)
    print(pro_pack)
    factor_data = df.iloc[:, df_column['x']].values
    print(f"x0**2 = {np.sum(factor_data[:, 0] * factor_data[:, 0])}")
    print(f"x1**2 = {np.sum(factor_data[:, 1] * factor_data[:, 1])}")
    print(f"x0*x1 = {np.sum(factor_data[:, 0] * factor_data[:, 1])}")
    df_val, _ = orth_panel_global_residual(df_val, df_column, column=df_column['x'], **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_panel_local_residual():
    df, df_column = _get_fX_df(rest=6)
    print(df)
    df_val = copy.deepcopy(df)
    top_k = None
    df, pro_pack = orth_panel_local_residual(df, df_column, df_column['x'][:2], df_column['x'][2:])
    print(df)
    print(pro_pack)
    factor_data = df.iloc[:, df_column['x']].values
    print(f"x0**2 = {np.sum(factor_data[:, 0] * factor_data[:, 0])}")
    print(f"x1**2 = {np.sum(factor_data[:, 1] * factor_data[:, 1])}")
    print(f"x0*x1 = {np.sum(factor_data[:, 0] * factor_data[:, 1])}")
    df_val, _ = orth_panel_local_residual(df_val, df_column, df_column['x'][:2], **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_cross_global_symmetry():
    df, df_column = _get_fX_df(rest=3)
    df_val = copy.deepcopy(df)
    print(df)
    df, pro_pack = orth_cross_global_symmetry(df, df_column, column=df_column['x'], verbose=True)
    print(df)
    df_val, _ = orth_cross_global_symmetry(df_val, df_column, column=df_column['x'], **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_cross_global_standard():
    df, df_column = _get_fX_df(rest=3)
    df_val = copy.deepcopy(df)
    print(df)
    df, pro_pack = orth_cross_global_standard(df, df_column, column=df_column['x'], verbose=True)
    print(df)
    df_val, _ = orth_cross_global_standard(df_val, df_column, column=df_column['x'], **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_cross_global_custom():
    df, df_column = _get_fX_df(rest=3)
    df_val = copy.deepcopy(df)
    A = np.random.rand(3, 3)
    U, S, V = np.linalg.svd(A)
    custom = np.dot(U, V)
    print(custom)
    print(np.dot(custom.T, custom))
    print(df)
    df, pro_pack = orth_cross_global_custom(df, df_column, column=df_column['x'], custom_transform=custom)
    print(df)
    df_val, _ = orth_cross_global_custom(df_val, df_column, column=df_column['x'], **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_cross_global_residual():
    df, df_column = _get_fX_df(rest=3)
    print(df)
    df_val = copy.deepcopy(df)
    top_k = None
    df, pro_pack = orth_cross_global_residual(df, df_column, column=df_column['x'])
    print(df)
    print(pro_pack)
    factor_data = df.iloc[:, df_column['x']].values
    print(f"x0**2 = {np.sum(factor_data[:, 0] * factor_data[:, 0])}")
    print(f"x1**2 = {np.sum(factor_data[:, 1] * factor_data[:, 1])}")
    print(f"x0*x1 = {np.sum(factor_data[:, 0] * factor_data[:, 1])}")
    df_val, _ = orth_cross_global_residual(df_val, df_column, column=df_column['x'], **pro_pack)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_orth_cross_local_residual():
    df, df_column = _get_fX_df(rest=6)
    print(df)
    df_val = copy.deepcopy(df)
    top_k = None
    df, pro_pack = orth_panel_local_residual(df, df_column, df_column['x'][:2], df_column['x'][2:])
    df = reduce_mem_usage_df(df, df_column['x'])
    print(df)
    print(pro_pack)
    factor_data = df.iloc[:, df_column['x']].values
    print(f"x0**2 = {np.sum(factor_data[:, 0] * factor_data[:, 0])}")
    print(f"x1**2 = {np.sum(factor_data[:, 1] * factor_data[:, 1])}")
    print(f"x0*x1 = {np.sum(factor_data[:, 0] * factor_data[:, 1])}")
    df_val, _ = orth_panel_local_residual(df_val, df_column, df_column['x'][:2], **pro_pack)
    df_val = reduce_mem_usage_df(df_val, df_column['x'])
    assert (df_val.values == df.values).all()
    print("success.")


if __name__ == "__main__":
    # debug_orth_panel_global_symmetry()
    # debug_panel_orth_global_standard()
    # debug_panel_orth_global_custom()
    # debug_panel_orth_global_residual()
    # debug_panel_orth_local_residual()
    # debug_orth_cross_global_symmetry()
    # debug_orth_cross_global_standard()
    # debug_orth_cross_global_custom()
    # debug_orth_cross_global_residual()
    debug_orth_cross_local_residual()