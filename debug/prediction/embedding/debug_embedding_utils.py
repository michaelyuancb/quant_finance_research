from quant_finance_research.utils import get_example_cat_matrix, get_example_cat_df
from quant_finance_research.eda.eda_feat import eda_data_type
from quant_finance_research.prediction.embedding.embedding_utils import *


def debug_category2int_global():
    df, df_column = get_example_cat_df()
    dict_abs, dict_rel, dfr = eda_data_type(df, df_column, print_result=False)
    cat_column = dict_abs['x_object'] + dict_abs['x_int']
    print(df)
    print(cat_column)
    df, total_cat_num = category2int_global(df, df_column, cat_column)
    print(df)
    print(total_cat_num)


if __name__ == "__main__":
    debug_category2int_global()
