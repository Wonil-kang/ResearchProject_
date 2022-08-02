import joblib
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from tensorflow import metrics

import LSTM
import Loger
import datetime

import Math
import loadData


file_path_LSTM = 'log/result_importance_svm.log'

mse_min = 100.0
rmse_min = 100.0
mae_min = 100.0
acc_max = 0.0


def run_SVM(df, market_type):

    # Normalize
    df = LSTM.normalize(df)

    feature_cols_us, feature_cols = loadData.get_feature_cols(market_type)
    label_cols = ['index_' + market_type]

    train_df = df[0:LSTM.split]
    test_df = df[LSTM.split:]

    train_df = sklearn.utils.shuffle(train_df)

    train_feature_df_us = pd.DataFrame(train_df, columns=feature_cols_us)
    train_feature_df = pd.DataFrame(train_df, columns=feature_cols)
    train_label_df = pd.DataFrame(train_df, columns=label_cols)

    test_feature_df_us = pd.DataFrame(test_df, columns=feature_cols_us)
    test_feature_df = pd.DataFrame(test_df, columns=feature_cols)
    test_label_df = pd.DataFrame(test_df, columns=label_cols)

    # Change to Numpy
    x_train_us = train_feature_df_us.to_numpy()
    x_train = train_feature_df.to_numpy()
    y_train = train_label_df.to_numpy()

    x_test_us = test_feature_df_us.to_numpy()
    x_test = test_feature_df.to_numpy()
    y_test = test_label_df.to_numpy()

    do(market_type, x_train, y_train, x_test, y_test, 'non_us')
    do(market_type, x_train_us, y_train, x_test_us, y_test, 'us')


def do(market_type, x_train, y_train, x_test, y_test, data_type):

    reset_record()


    for thisGamma in [.00001, .00005, .0001, .0002, .0003, .0004, .0005, .0006, .0007, .0008, .0009, 0.001]:
        for thisC in [1, 5, 10, 20, 40, 100, 1000, 10000]:

            model = SVR(kernel="rbf", C=thisC,
                         gamma=thisGamma).fit(x_train, y_train.ravel())

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

            final_log = market_type + ",SVM," + data_type + "," + \
                        str(thisGamma) + "," + str(thisC) + "," + \
                        str(mse) + "," + str(rmse) + "," + str(mae) + "," + str(accuracy) + "," + result_file_path

            Loger.log(final_log)


def reset_record():
    global mae_min, rmse_min, mse_min, acc_max

    mse_min = 100.0
    rmse_min = 100.0
    mae_min = 100.0
    acc_max = 0.0