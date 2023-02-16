# Quantative Finance Research Harbour
Some Quantative Finance Python Tool & Project For Research &amp; Application

This is a repository which contains things in my quantative finance research & exploration.

Created by Michael Yuan.
Contact Me with michael.yuan.cb@whu.edu.cn.

Some useful tool and repository are also listed below:
 - [MindQuantFinance](https://gitee.com/luweizheng/mind-quant-finance) & [MindTimeSeriesSIG](https://gitee.com/mindspore/community/tree/master/sigs/TimeSequence)
 - [TSCV: Extension TimeSeries Dataset Split Tool](https://github.com/WenjieZ/TSCV)

<image src="src/img/qfr_fold.jpg" width="500" height="180">

<br>

# User Guide For Quantative Finance Research Library:

Usage & Improvement are welcome. :)

Because of time reason, I haven't written the user document for this library, but this will be supplemented later. Some core descriptions are listed below to help starting.

This library can also be used in None-Financial Area whenever it is about dealing with tabular data, especially when the data is non-stationary.

QwQ

Core Data Structure:

 - data_df:  pandas.DataFrame with the 1st and 2nd column be ["time_id", "investment_id"]. Other columns contain "x_column", "y_column" and "loss_column". (loss_column is used for calculation of loss function.)

 - df_column: a dict to point out the [x_column, y_column, loss_column]. Format: df_column={"x": x_column, "y": y_column, "loss": loss_column}. Each column is a list with interger as its items.

Core Framework & Function:

 - preprocess_function: With input as (data_df, df_column, column, other_parameters, **kwargs). Its ouput will be (data_df, pro_package). "pro_package" is a dict to reproduce the preprocess to out-sample data. To reproduce the preprocess, use preprocess_function(outsample_df, outsample_df_column, column, **pro_package). Notice that if "df_column_new" in pro_package.keys(), it means that the preprocess change the column of DataFrame. One can use df_column = update_df_column_package(df_column, pro_package) [in fe.fe_utils] to get the new df_column for later usage. Class DebugPiplinePreprocess can help you build your own preprocess pipeline.

 - tscv: The cross-validation tools for TimeSeries Finanial Data. QuantTimeSplit_PreprocessSplit can combine the preprocess with QuantTimeSplit.

 - strategy: The framework for training & prediction. Notice that for training & prediction, the data_df(or train_df & val_df) and df_column will always be the first two parameters. right now it only supports cross-sectional data, but time-series and panel data will be supported later.

 Before the updatation of user document, A more detailed usage could be seen in debug file.