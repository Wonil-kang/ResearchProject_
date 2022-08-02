import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.python.keras.layers import RNN, SimpleRNN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
import datetime

import LSTM
import Loger
import loadData

mse_min = 100.0
rmse_min = 100.0
mae_min = 100.0
acc_max = 0.0

def run_RNN(df, market_type):
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

    reset_record()

    for layer_unit in [64, 128, 256, 512, 1024]:
        for windows in [10, 20, 30, 40, 50, 60]:
            for act in ['tanh', 'sigmoid', 'relu', 'softsign', 'softmax']:
                run(market_type, feature_np, label_np, feature_cols_type, layer_unit, act, windows)


def run(market_type, feature_np, label_np, feature_cols_type, layer_unit, activation_functions, window_size):

    X, Y = LSTM.make_sequence_dataset(feature_np, label_np, window_size)
    # print(X.shape, Y.shape)

    x_train = X[0:LSTM.split]
    y_train = Y[0:LSTM.split]

    x_test = X[LSTM.split:]
    y_test = Y[LSTM.split:]

    model = Sequential()

    model = Sequential()
    model.add(SimpleRNN(layer_unit, activation=activation_functions, input_shape=x_train[0].shape))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=64, callbacks=[early_stop])

    # Get Prediction
    y_predict = model.predict(x_test)

    mse = metrics.mean_squared_error(y_test, y_predict)
    rmse = metrics.mean_squared_error(y_test, y_predict) ** 0.5
    mae = metrics.mean_absolute_error(y_test, y_predict)

    accuracy = get_accuracy(y_test, y_predict)

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
        print('Change Record : ' + str(mse_min) + "," + str(rmse_min) + "," + str(mae_min) + "," + str(
            acc_max))

        # Save Model
        now = datetime.datetime.now()
        dateFormat = '%Y%m%d%H%M%S'
        str_datetime = datetime.datetime.strftime(now, dateFormat)

        result_file_path = "modelResults/" + str_datetime + ".hs"
        model.save(result_file_path)

    final_log = market_type + ",RNN," + feature_cols_type + "," + str(layer_unit) + "," + str(
        activation_functions) + "," + \
                str(window_size) + "," + str(mse) + "," + str(rmse) + "," + str(mae) + "," + str(
        accuracy) + "," + result_file_path

    Loger.log(final_log)



def load_RNN(file_path, feature_np, label_np, window_size):

    X, Y = make_sequence_dataset(feature_np, label_np, window_size)
    # print(X.shape, Y.shape)

    x_test = X[LSTM.split:]
    y_test = Y[LSTM.split:]

    model = tf.keras.models.load_model(file_path)
    model.summary()

    y_predict = model.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_predict) * 100

    plotResultValue(file_path, y_test, y_predict)


def plotResultValue(title, y_test, prediction):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel('period')
    plt.ylabel('value')
    plt.ylabel('value')
    plt.plot(y_test, label='actual')
    plt.plot(prediction, label='prediction')
    plt.grid()
    plt.legend(loc='best')

    plt.show()





def showChart(df):
    fig, ax1 = plt.subplots()
    ax1.set_title('Asian Stock Indexes', fontsize=16)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value', color='b')
    ax1.plot(df['Date'], df['nk_price'], label='NIKKEI', marker='s', color='b')
    ax1.plot(df['Date'], df['hs_price'], label='HANGSENG', marker='s', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Value', color='r')
    ax2.plot(df['Date'], df['kp_price'], label='KOSPI', marker='s', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    plt.show()


def showKoreanChart(df):
    plt.figure(figsize=(7, 4))

    plt.title('Asian Stock Indexes')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.plot(df['kp_price'], label='KOSPI', color='b')
    plt.plot(df['nk_price'], label='NIKKEI', color='r')
    plt.plot(df['hs_price'], label='HANGSENG', color='p')
    plt.legend(loc='best')
    plt.show()


def normalize(df):
    scaler = MinMaxScaler()

    scale_cols = ['open_s100', 'high_s100', 'low_s100',
                  'close_s100', 'adj_close_s100', 'volume_s100', 'day', 'open_n100',
                  'high_n100', 'low_n100', 'close_n100', 'adj_close_n100', 'volume_n100',
                  'open_dw', 'high_dw', 'low_dw', 'close_dw', 'adj_close_dw', 'volume_dw',
                  'kp_price', 'nk_price', 'hs_price', 'open_kp', 'high_kp',
                  'low_kp', 'close_kp', 'adj_close_kp', 'volume_kp', 'open_nk', 'high_nk',
                  'low_nk', 'close_nk', 'adj_close_nk', 'volume_nk', 'open_hs', 'high_hs',
                  'low_hs', 'close_hs', 'adj_close_hs', 'volume_hs']

    scaled_df = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)
    return scaled_df


# Standardize
def standardize(df, value):
    scaler = StandardScaler()

    scaler.fit(df[label_cols])
    rst_scaled = scaler.transform(value)
    rst_df_scaled = pd.DataFrame(data=rst_scaled)

    return rst_df_scaled


def get_accuracy(y_test, y_predict):
    length = len(y_test)

    accuracy_sum = 0

    for i in range(y_test.shape[0]):
        error = y_test[i] - y_predict[i]

        if error < 0:
            error = -error

        accuracy = (y_test[i] - error) / y_test[i] * 100
        accuracy_sum += accuracy

    average_accuracy = accuracy_sum / length

    return average_accuracy[0]



def reset_record():
    global mae_min, rmse_min, mse_min, acc_max

    mse_min = 100.0
    rmse_min = 100.0
    mae_min = 100.0
    acc_max = 0.0
    acc_max = 0.0