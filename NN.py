import Math
import pandas as pd
import sklearn
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn import metrics

import LSTM
import Loger
import datetime

import loadData


mse_min = 100.0
rmse_min = 100.0
mae_min = 100.0
acc_max = 0.0


def run_NN(df, market_type):
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

    for l1 in [64, 128, 256, 512, 1024]: # 64, 128,
        for l2 in [64, 128, 256, 512, 1024]:

            if l1 > l2:
                continue

            for l3 in [64, 128, 256, 512, 1024]:

                if l2 > l3:
                    continue

                for act1 in ['tanh', 'sigmoid', 'relu','softsign', 'softmax']:
                    for act2 in ['tanh', 'sigmoid', 'relu','softsign', 'softmax']:
                        for act3 in ['tanh', 'sigmoid', 'relu', 'softsign', 'softmax']:

                            model = Sequential()
                            model.add(Dense(l1, activation=act1))
                            model.add(Dense(l2, activation=act2))
                            model.add(Dense(l3, activation=act3))
                            model.add(Dense(1, activation='linear'))

                            model.compile(loss='mse', optimizer='adam', metrics=['mae'])

                            early_stop = EarlyStopping(monitor='val_loss', patience=5)

                            model.fit(x_train, y_train, validation_data=(x_test, y_test),
                                      epochs=1000, batch_size=64, callbacks=[early_stop])

                            y_predict = model.predict(x_test)

                            mse = metrics.mean_squared_error(y_test, y_predict)
                            rmse = metrics.mean_squared_error(y_test, y_predict) ** 0.5
                            mae = metrics.mean_absolute_error(y_test, y_predict)

                            accuracy = LSTM.get_accuracy(y_test, y_predict)

                            statistics_result = str(mse) + "," + str(rmse) + "," + str(mae) + "," + str(accuracy)
                            print(statistics_result)

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
                                print('Change Record : ' + str(mse_min) + "," + str(rmse_min) + "," + str(
                                    mae_min) + "," + str(acc_max))

                                # Save Model
                                now = datetime.datetime.now()
                                dateFormat = '%Y%m%d%H%M%S'
                                str_datetime = datetime.datetime.strftime(now, dateFormat)

                                result_file_path = "modelResults/" + str_datetime + ".hs"
                                model.save(result_file_path)

                            final_log = market_type + ",NN," + data_type + "," + \
                                        str(l1) + "," + str(l2) + "," + str(l3) + "," + \
                                        str(act1) + "," + str(act2) + "," + str(act3) + "," + \
                                        str(mse) + "," + str(rmse) + "," + str(mae) + "," + \
                                        str(accuracy) + "," + result_file_path

                            Loger.log(final_log)


def reset_record():
    global mae_min, rmse_min, mse_min, acc_max

    mse_min = 100.0
    rmse_min = 100.0
    mae_min = 100.0
    acc_max = 0.0