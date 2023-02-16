
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
        pipeline.add_feat_preprocess(delete_NanRatio_Feature)
        pipeline.add_feat_preprocess(add_GlobalAbsIcRank_LocalTimeMeanFactor)
        df, df_column = get_example_df()
        for j in range(8):
            df.iloc[j, 2] = np.nan
        column = df_column['x']
        print(df)
        df2, pro = pipeline(df, df_column, column, del_nan_ratio=0.3, number_GAIR=1, time_col='time_id')
        print(df2)
        print(pro)
        df3, _ = pipeline(df, df_column, column, **pro)
        assert (df3.values == df2.values).all()
        print("success.")


if __name__ == "__main__":
    # debug_update_df_column_package()
    DebugPiplinePreprocess().debug_combination_feat()