import Math
import pandas as pd
from sklearn.linear_model import LinearRegression

import LSTM
import Loger
import datetime

import loadData
from load_result import logResult, file_path_LSTM, file_path_LR, get_string


def run_LR(df, market_type):
    # Normalize
    df = LSTM.normalize(df)

    feature_cols_us, feature_cols = loadData.get_feature_cols(market_type)
    label_cols = ['index_' + market_type]

    feature_df_us = pd.DataFrame(df, columns=feature_cols_us)
    feature_df = pd.DataFrame(df, columns=feature_cols)
    label_df = pd.DataFrame(df, columns=label_cols)

    # Change to Numpy
    feature_np_us = feature_df_us.to_numpy()
    feature_np = feature_df.to_numpy()
    label_np = label_df.to_numpy()

    do(market_type, feature_np, label_np, 'non_us')
    do(market_type, feature_np_us, label_np, 'us')


def do(market_type, feature_np, label_np, feature_cols_type):
    x_train = feature_np[0:LSTM.split]
    y_train = label_np[0:LSTM.split]

    x_test = feature_np[LSTM.split:]
    y_test = label_np[LSTM.split:]

    model = LinearRegression()
    model.fit(x_train, y_train)

    print(model.intercept_)
    print(model.coef_)

    y_predict = model.predict(x_test)

    # Statistics Result
    mse = Math.mse(y_test, y_predict)
    rmse = Math.mse(y_test, y_predict) ** 0.5
    mae = Math.mae(y_test, y_predict)
    accuracy = LSTM.get_accuracy(y_test, y_predict)

    statistics_result = str(mse) + "," + str(rmse) + "," + str(mae) + "," + str(accuracy)
    print(statistics_result)

    # Save Model
    now = datetime.datetime.now()
    dateFormat = '%Y%m%d%H%M%S'
    str_datetime = datetime.datetime.strftime(now, dateFormat)

    result_file_path = "modelResults/" + str_datetime + ".hs"
    # model.save(result_file_path)

    final_log = market_type + ",LR," + feature_cols_type + "," + \
                str(mse) + "," + str(rmse) + "," + str(mae) + "," + str(accuracy) + "," + result_file_path

    Loger.log(final_log)

    # LSTM.plotResultValue("LR" + "-" + str(feature_cols_type), y_test, y_predict)


    rst_test = []
    rst_predict = []

    for i in range(len(y_test)):
        rst_test.append(y_test[i])
        rst_predict.append(y_predict[i])

    logResult(file_path_LR, 'LR,' + market_type + ',' + feature_cols_type)
    logResult(file_path_LR, get_string(rst_test))
    logResult(file_path_LR, get_string(rst_predict))
