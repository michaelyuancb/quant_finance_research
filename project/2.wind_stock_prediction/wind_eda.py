from quant_finance_research.utils import load_pickle, save_pickle
from quant_finance_research.eda.eda_feat import eda_nan_analysis, eda_factor_ic_analysis
from quant_finance_research.fe.fe_feat import delete_NanRatio_Feature

project_name = '1d'  # '1d or 10d'

data = load_pickle('wind_factor_insample.pkl')

data_columns = data.columns.tolist()
print(data_columns)

data = data.reset_index()
del data['MicroSecondSinceEpoch'], data['positive']
data = data.rename(columns={'time': 'time_id', 'YID': 'investment_id'})
del data['id']

x_column = [i for i in range(5, data.shape[1])]
loss_column = []
y_column = [3] if project_name == '1d' else [4]
df_column = {'x': x_column, 'y': y_column, 'loss': loss_column}

print("===================== Feature NaN Analysis=========================")
col_nan_ratio, row_nan_ratio = eda_nan_analysis(data, df_column, df_column['x'])
print("===================== Target NaN Analysis=========================")
col_nan_ratio, row_nan_ratio = eda_nan_analysis(data, df_column, df_column['y'])
data = data.dropna(axis=0, how='any')
print("===================== Target Drop-NaN Analysis=========================")
col_nan_ratio, row_nan_ratio = eda_nan_analysis(data, df_column, df_column['y'])

print("===================== Item EDA =========================")
print(f"number_time_id={data.time_id.nunique()} ; min={data.time_id.min()} ; max={data.time_id.max()}")
print(f"time_id_type={type(data['time_id'].iloc[0])}")
print(f"number_inv_id={data.investment_id.nunique()} ; min={data.investment_id.min()} ; max={data.investment_id.max()}")
print(f"inv_id_type={type(data['investment_id'].iloc[0])}")

print("===================== IC Factor EDA =======================")
df_ic_eda = eda_factor_ic_analysis(data, df_column)
print(df_ic_eda)

print("===================== IC Factor EDA =======================")
print(data.describe())