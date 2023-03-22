import copy
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression


def orth_choose_eigen(eig_vals, eig_vecs, top_k=None, eps_bar=1e-6):
    """
        Delete the negative eigen value and its engen vector for calculation stablity.
    """
    eig_vals = np.real_if_close(eig_vals)
    eig_vecs = np.real_if_close(eig_vecs)
    ps_pos = -1
    for i in range(eig_vals.shape[0]):
        if eig_vals[i] > eps_bar:
            ps_pos = i
            break
    if ps_pos != -1:
        eig_vals = eig_vals[ps_pos:]
        eig_vecs = eig_vecs[:, ps_pos:]
        if ps_pos > 0:
            print(f"{ps_pos} non-positive eigen-value is discarded. rest number of eigen: {eig_vals.shape[0]}")
    else:
        raise TypeError("The eigen-value of Corr is all non-positive, please check the property of input.")
    if top_k is not None and top_k <= eig_vals.shape[0]:
        eig_vecs = eig_vecs[:, -top_k:]
        eig_vals = eig_vals[-top_k:]
    return eig_vals, eig_vecs  # (k, )  , (N, k)


def orth_panel_global_symmetry(df_org, df_column, column, top_k=None, eps_bar=1e-6, orth_matrix=None, inplace=False,
                               **kwargs):
    """
        S = O * D^{-1/2} * O
    """
    df = df_org if inplace else copy.deepcopy(df_org)
    factor_data = df.iloc[:, column].values
    if orth_matrix is None:
        cov_mat = np.dot(factor_data.T, factor_data)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        eig_vals, eig_vecs = orth_choose_eigen(eig_vals, eig_vecs, top_k=top_k, eps_bar=eps_bar)
        orth_matrix = np.dot(np.dot(eig_vecs, np.diag(eig_vals ** (-0.5))), eig_vecs.T)
    factor_data = np.dot(factor_data, orth_matrix)
    df.iloc[:, column] = factor_data
    return df, {"orth_matrix": orth_matrix}


def orth_panel_global_standard(df_org, df_column, column, top_k=None, eps_bar=1e-6, orth_matrix=None, inplace=False,
                               **kwargs):
    """
        S = O * D^{-1/2}
    """
    df = df_org if inplace else copy.deepcopy(df_org)
    factor_data = df.iloc[:, column].values
    if orth_matrix is None:
        cov_mat = np.dot(factor_data.T, factor_data)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        eig_vals, eig_vecs = orth_choose_eigen(eig_vals, eig_vecs, top_k=top_k, eps_bar=eps_bar)
        if eig_vals.shape[0] < len(column):
            raise ValueError(f"The number of valid eigenvalue is {eig_vals.shape[0]}, while the len(column)="
                             f"{len(column)}, which is less then the number(eigenvalue).")
        orth_matrix = np.dot(eig_vecs, np.diag(eig_vals ** (-0.5)))
    factor_data = np.dot(factor_data, orth_matrix)
    df.iloc[:, column] = factor_data
    return df, {"orth_matrix": orth_matrix}


def orth_panel_global_custom(df_org, df_column, column, custom_transform=None, top_k=None, eps_bar=1e-6, inplace=False,
                             orth_matrix=None, **kwargs):
    """
        S = O * D^{-1/2} * O * custom_transform.
        when custom_transform is None, it is equal to orth_global_symmetry()
    """
    df = df_org if inplace else copy.deepcopy(df_org)
    factor_data = df.iloc[:, column].values
    if orth_matrix is None:
        if custom_transform is not None:
            if custom_transform.shape[0] != factor_data.shape[1]:
                raise ValueError(f"The custom_transform.shape[0]({custom_transform.shape[0]}) != "
                                 f"factor_data.shape[1]({factor_data.shape[1]}).")
            if not np.allclose(np.dot(custom_transform.T, custom_transform), np.eye(custom_transform.shape[0])):
                raise ValueError("The custom_transform provided is not orthogonal matrix.")
        cov_mat = np.dot(factor_data.T, factor_data)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        eig_vals, eig_vecs = orth_choose_eigen(eig_vals, eig_vecs, top_k=top_k, eps_bar=eps_bar)
        orth_matrix = np.dot(np.dot(eig_vecs, np.diag(eig_vals ** (-0.5))), eig_vecs.T)
        orth_matrix = np.dot(orth_matrix, custom_transform)
    factor_data = np.dot(factor_data, orth_matrix)
    df.iloc[:, column] = factor_data
    return df, {"orth_matrix": orth_matrix}


def orth_panel_global_residual(df_org, df_column, column, orth_coef_list=None, inplace=False,
                               **kwargs):
    """
        perform the orthogonalization by linear regression (alternative with residual).
        orth_coef: List(array),  the shape of array is [1, 1], [1, 2], [1, 3] ... [1, N-2], [1, N-1].
    """
    df = df_org if inplace else copy.deepcopy(df_org)
    factor_data = df.iloc[:, column].values
    n_column = len(column)
    if orth_coef_list is None:
        orth_coef_list = []
        for i in range(1, n_column):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(factor_data[:, :i], factor_data[:, i])
            factor_data[:, i] = factor_data[:, i] - reg.predict(factor_data[:, :i])
            orth_coef_list.append(reg.coef_.reshape(1, -1))
    else:
        for i in range(1, n_column):
            factor_data[:, i] = factor_data[:, i] - np.sum(orth_coef_list[i - 1] * factor_data[:, :i], axis=1)
            # orth_coef: np.array[1, len]
    df.iloc[:, column] = factor_data
    return df, {"orth_coef_list": orth_coef_list}


def orth_panel_local_residual(df_org, df_column, column, orth_column_base, orth_coef_list=None, inplace=False,
                              **kwargs):
    """
        perform the orthogonalization by linear regression (alternative with residual).
        local means that we only use column_base to regress the column and get the column's residual.
    """
    df = df_org if inplace else copy.deepcopy(df_org)
    if len(list(set(column + orth_column_base))) != len(column) + len(orth_column_base):
        raise ValueError("The column[target] and column_base should not be overlapping.")
    factor_target = df.iloc[:, column].values
    factor_base = df.iloc[:, orth_column_base].values
    n_column = len(column)
    if orth_coef_list is None:
        orth_coef_list = []
        for i in range(1, n_column):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(factor_base, factor_target[:, i])
            factor_target[:, i] = factor_target[:, i] - reg.predict(factor_base)
            orth_coef_list.append(reg.coef_.reshape(1, -1))
    else:
        for i in range(1, n_column):
            factor_target[:, i] = factor_target[:, i] - np.sum(orth_coef_list[i - 1] * factor_base, axis=1)
            # orth_coef: np.array[1, len]
    df.iloc[:, column] = factor_target
    return df, {"orth_coef_list": orth_coef_list, "orth_column_base": orth_column_base}


def orth_cross_global_symmetry(df_org, df_column, column, time_col='time_id', top_k=None, eps_bar=1e-6, verbose=False,
                               **kwargs):
    """
        S = O * D^{-1/2} * O, for cross section.
    """
    df = copy.deepcopy(df_org)
    df_time_idx = df.groupby(time_col).count().index
    for i in range(len(df_time_idx)):
        mask = df[time_col] == df_time_idx[i]
        if df[mask].shape[0] < len(column):
            if verbose:
                print(f"skip orth {df_time_idx[i]}; only {df[mask].shape[0]} data point is observed, "
                      f"while the len(column)={len(column)}.")
            continue
        df[mask], _ = orth_panel_global_symmetry(df[mask].copy(), df_column, column, top_k, eps_bar, inplace=True)
    return df, {"time_col": time_col, "top_k": top_k, "eps_bar": eps_bar}


def orth_cross_global_standard(df_org, df_column, column, time_col='time_id', top_k=None, eps_bar=1e-6, verbose=False,
                               **kwargs):
    """
        S = O * D^{-1/2}, for cross section.
    """
    df = copy.deepcopy(df_org)
    df_time_idx = df.groupby(time_col).count().index
    for i in range(len(df_time_idx)):
        mask = df[time_col] == df_time_idx[i]
        if df[mask].shape[0] < len(column):
            if verbose:
                print(f"skip orth {df_time_idx[i]}; only {df[mask].shape[0]} data point is observed, "
                      f"while the len(column)={len(column)}.")
            continue
        df[mask], _ = orth_panel_global_standard(df[mask].copy(), df_column, column, top_k, eps_bar, inplace=True)
    return df, {"time_col": time_col, "top_k": top_k, "eps_bar": eps_bar}


def orth_cross_global_custom(df_org, df_column, column, time_col='time_id', custom_transform=None, top_k=None,
                             eps_bar=1e-6, verbose=False, **kwargs):
    """
        S = O * D^{-1/2} * O * custom_transform., for cross section.
    """
    df = copy.deepcopy(df_org)
    df_time_idx = df.groupby(time_col).count().index
    for i in range(len(df_time_idx)):
        mask = df[time_col] == df_time_idx[i]
        if df[mask].shape[0] < len(column):
            if verbose:
                print(f"skip orth {df_time_idx[i]}; only {df[mask].shape[0]} data point is observed, "
                      f"while the len(column)={len(column)}.")
            continue
        df[mask], _ = orth_panel_global_custom(df[mask].copy(), df_column, column, custom_transform, top_k, eps_bar,
                                               inplace=True)
    return df, {"time_col": time_col, "top_k": top_k, "eps_bar": eps_bar, "custom_transform": custom_transform}


def orth_cross_global_residual(df_org, df_column, column, time_col='time_id', verbose=False,
                               **kwargs):
    """
        perform the orthogonalization by linear regression (alternative with residual).
        for cross section.
    """
    df = copy.deepcopy(df_org)
    df_time_idx = df.groupby(time_col).count().index
    for i in range(len(df_time_idx)):
        mask = df[time_col] == df_time_idx[i]
        df[mask], _ = orth_panel_global_residual(df[mask].copy(), df_column, column, inplace=True)
    return df, {"time_col": time_col}


def orth_cross_local_residual(df_org, df_column, column, orth_column_base, time_col='time_id', verbose=False,
                              **kwargs):
    """
        perform the orthogonalization by linear regression (alternative with residual).
        local means that we only use column_base to regress the column and get the column's residual.
        for cross section.
    """
    df = copy.deepcopy(df_org)
    df_time_idx = df.groupby(time_col).count().index
    for i in range(len(df_time_idx)):
        mask = df[time_col] == df_time_idx[i]
        df[mask], _ = orth_panel_local_residual(df[mask].copy(), df_column, column, orth_column_base, inplace=True)
    return df, {"time_col": time_col, "orth_column_base": orth_column_base}
