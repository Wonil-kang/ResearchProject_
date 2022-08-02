import Math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

import LSTM
import Loger
import datetime

import loadData

mse_min = 100.0
rmse_min = 100.0
mae_min = 100.0
acc_max = 0.0

def run_RF(df, market_type):
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


def load(df, file_path):
    model = joblib.load(file_path)


def do(market_type, feature_np, label_np, feature_cols_type):

    reset_record()

    x_train = feature_np[0:LSTM.split]
    y_train = label_np[0:LSTM.split]

    x_test = feature_np[LSTM.split:]
    y_test = label_np[LSTM.split:]

    bootstrap = [True, False]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]
    max_features = ['auto', 'sqrt']
    n_estimators = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    for bs in bootstrap:  # 64, 128,
        for md in max_depth:
            for mf in max_features:
                for ne in n_estimators:
                    model = RandomForestRegressor(
                        n_estimators=ne,
                        random_state=50,  # 42
                        min_samples_split=2,
                        min_samples_leaf=50,
                        max_features=mf,
                        max_depth=md,
                        bootstrap=bs)

                    model.fit(x_train, y_train.ravel())

                    y_predict = model.predict(x_test)
                    mse = Math.mse(y_test, y_predict)
                    rmse = Math.mse(y_test, y_predict) ** 0.5
                    mae = Math.mae(y_test, y_predict)
                    accuracy = LSTM.get_accuracy(y_test, y_predict)

                    result_file_path = '-'
                    save_flag = False
                    global mae_min, rmse_min, mse_min, acc_max

                    if mse < mse_min:
                        mse_min = mse
                        save_flag = True

                    if rmse < rmse_min:
                        rmse_min = rmse
                        save_flag = True

                    if mae < mae_min:
                        mae_min = mae
                        save_flag = True

                    if accuracy > acc_max:
                        acc_max = accuracy
                        save_flag = True

                    if save_flag:
                        print('Change Record : ' + str(mse_min) + "," + str(rmse_min) + "," + str(mae_min) + "," + str(
                            acc_max))

                        # Save Model
                        now = datetime.datetime.now()
                        dateFormat = '%Y%m%d%H%M%S'
                        str_datetime = datetime.datetime.strftime(now, dateFormat)

                        result_file_path = "modelResults/" + str_datetime + ".hs"
                        # model.save(result_file_path)
                        joblib.dump(model, result_file_path, compress=0)

                    final_log = market_type + ",RF," + feature_cols_type + "," + \
                                str(bs) + "," + str(md) + "," + str(mf) + "," + str(ne) + "," + \
                                str(mse) + "," + str(rmse) + "," + str(mae) + "," + str(accuracy) + "," + result_file_path

                    Loger.log(final_log)


def reset_record():
    global mae_min, rmse_min, mse_min, acc_max

    mse_min = 100.0
    rmse_min = 100.0
    mae_min = 100.0
    acc_max = 0.0
