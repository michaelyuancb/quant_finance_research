import copy

from quant_finance_research.fe.fe_portfolio import *
from quant_finance_research.fe.fe_val import *
from quant_finance_research.fe.fe_orth import *
from quant_finance_research.fe.fe_feat import *
from quant_finance_research.fe.fe_utils import *
from quant_finance_research.utils import *


def debug_update_df_column_package():
    df, df_column = get_example_df()
    print(df)
    df2, pro = add_GlobalAbsIcRank_LocalTimeMeanFactor(df, df_column['x'], df_column, number_GAIR=1, time_col="time_id")
    print(df2)
    df_column = update_df_column_package(df_column, pro)
    print(df_column)


class DebugPiplinePreprocess:

    def debug_combination_feat(self):
        pipeline = PiplinePreprocess()
        df, df_column = get_example_df()
        column = df_column['x']
        pipeline.add_feat_preprocess(delete_NanRatio_Feature, df_column, column)
        pipeline.add_feat_preprocess(add_GlobalAbsIcRank_LocalTimeMeanFactor, df_column, column)
        for j in range(8):
            df.iloc[j, 2] = np.nan
        print(df)
        df2, pro = pipeline(df, df_column, column, del_nan_ratio=0.3, number_GAIR=1, time_col='time_id')
        print(df2)
        print(pro)
        df3, _ = pipeline(df, df_column, column, **pro)
        assert (df3.values == df2.values).all()
        print("success.")


def debug_add_preprocess_pipeline():
    df, df_column = get_example_df()
    preprocess_func = PiplinePreprocess()
    df_val = copy.deepcopy(df)
    df, df_column, pro_pack, prop_return = add_preprocess_pipeline(preprocess_func, orth_panel_local_residual, df,
                                                                   df_column,
                                                                   df_column['x'][:1],
                                                                   orth_column_base=df_column['x'][1:])
    print(df.columns)
    print(prop_return)
    df, df_column, pro_pack, prop_return = add_preprocess_pipeline(preprocess_func,
                                                                   add_GlobalAbsIcRank_LocalTimeMeanFactor, df,
                                                                   df_column,
                                                                   df_column['x'], pro_pack, number_GAIR=1)
    print(df.columns)
    print(prop_return)
    df_column_org = copy.deepcopy(df_column)
    df, df_column, pro_pack, prop_return = add_preprocess_pipeline(preprocess_func, build_pca_cov_portfolio, df,
                                                                   df_column,
                                                                   df_column['x'], pro_pack, pca_cov_n_components='mle')
    print(df.columns)
    print(prop_return)
    print(df_column['x'])
    df, df_column, pro_pack, prop_return = add_preprocess_pipeline(preprocess_func, delete_Feature, df, df_column,
                                                                   df_column_org['x'], pro_pack)
    print(df.columns)
    print(prop_return)
    print("++++++++++++++++++++++++++++++++++++++++++++++<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(df.columns)
    print(df_column)
    print(df_column_org['x'])
    print(pro_pack)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    df, df_column = get_example_df()
    print(df)
    df, pro_package = preprocess_func(df, df_column, [], **pro_pack)
    print(df.columns)
    print(pro_package)
    print(df)
    df_val, _ = preprocess_func(df_val, df_column, df_column['x'], **pro_package)
    assert (df.values == df_val.values).all()
    print("success.")


if __name__ == "__main__":
    # debug_update_df_column_package()
    # DebugPiplinePreprocess().debug_combination_feat()
    debug_add_preprocess_pipeline()
