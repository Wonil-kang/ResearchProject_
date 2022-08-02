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
import fileIO
import loadData

file_path_RNN = 'log/result_rnn.log'
file_path_LSTM = 'log/result_lstm.log'
file_path_LR = 'log/result_LR.log'
file_path_ANN = 'log/result_ANN.log'


def load_RNN(df_kp, df_nk, df_hs):
    file_paths = ['modelResults/20220608095940.hs', 'modelResults/20220608095554.hs', 'modelResults/20220608143137.hs',
                  'modelResults/20220607143229.hs',
                  'modelResults/20220607175538.hs', 'modelResults/20220607230437.hs', 'modelResults/20220608004644.hs',
                  'modelResults/20220608085518.hs']

    windows = [30, 10, 30, 60, 60, 50, 10, 50]

    country_types = ['hs', 'hs', 'hs', 'kp', 'kp', 'nk', 'nk', 'nk']

    data_types = ['non_us', 'non_us', 'us', 'non_us', 'us', 'non_us', 'non_us', 'us']

    for i in range(len(file_paths)):
        if country_types[i] == 'hs':
            load_rnn_file(df_hs, country_types[i], data_types[i], file_paths[i], windows[i])

        elif country_types[i] == 'kp':
            load_rnn_file(df_kp, country_types[i], data_types[i], file_paths[i], windows[i])

        elif country_types[i] == 'nk':
            load_rnn_file(df_nk, country_types[i], data_types[i], file_paths[i], windows[i])


def load_rnn_file(df, market_type, data_type, file_path, window_size):
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

    x_test, y_test = [], []

    if data_type == 'non_us':
        X, Y = LSTM.make_sequence_dataset(feature_np, label_np, window_size)
        x_test = X[LSTM.split:]
        y_test = Y[LSTM.split:]

    else:
        X, Y = LSTM.make_sequence_dataset(feature_np_us, label_np, window_size)
        x_test = X[LSTM.split:]
        y_test = Y[LSTM.split:]

    model = tf.keras.models.load_model(file_path)
    model.summary()

    y_predict = model.predict(x_test)

    rst_test = []
    rst_predict = []

    for i in range(len(y_test)):
        rst_test.append(y_test[i])
        rst_predict.append(y_predict[i])

    logResult(file_path_RNN, file_path + "," + str(window_size))

    logResult(file_path_RNN, get_string(rst_test))
    logResult(file_path_RNN, get_string(rst_predict))


def load_LSTM(df_kp, df_nk, df_hs):
    file_paths = ['modelResults/20220604001456.hs', 'modelResults/20220604110724.hs', 'modelResults/20220604180657.hs', 'modelResults/20220604180825.hs',
                  'modelResults/20220605032022.hs', 'modelResults/20220605202935.hs', 'modelResults/20220605173706.hs',
                  'modelResults/20220606054438.hs', 'modelResults/20220606103357.hs', 'modelResults/20220607003046.hs','modelResults/20220607042255.hs']

    windows = [40, 50, 20, 20, 30, 40, 20, 50, 20, 30, 20]

    country_types = ['kp', 'kp', 'kp', 'kp', 'nk', 'nk', 'nk', 'hs', 'hs', 'hs', 'hs']

    data_types = ['non_us', 'non_us', 'us', 'us', 'non_us', 'us', 'us', 'non_us', 'non_us', 'us', 'us']

    for i in range(len(file_paths)):
        if country_types[i] == 'hs':
            load_LSTM_file(df_hs, country_types[i], data_types[i], file_paths[i], windows[i])

        elif country_types[i] == 'kp':
            load_LSTM_file(df_kp, country_types[i], data_types[i], file_paths[i], windows[i])

        elif country_types[i] == 'nk':
            load_LSTM_file(df_nk, country_types[i], data_types[i], file_paths[i], windows[i])


def load_LSTM_file(df, market_type, data_type, file_path, window_size):
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

    x_test, y_test = [], []

    if data_type == 'non_us':
        X, Y = LSTM.make_sequence_dataset(feature_np, label_np, window_size)
        x_test = X[LSTM.split:]
        y_test = Y[LSTM.split:]

    else:
        X, Y = LSTM.make_sequence_dataset(feature_np_us, label_np, window_size)
        x_test = X[LSTM.split:]
        y_test = Y[LSTM.split:]

    model = tf.keras.models.load_model(file_path)
    model.summary()

    y_predict = model.predict(x_test)

    rst_test = []
    rst_predict = []

    for i in range(len(y_test)):
        rst_test.append(y_test[i])
        rst_predict.append(y_predict[i])

    logResult(file_path_LSTM, 'LSTM,' + file_path + ',' + str(window_size))

    logResult(file_path_LSTM, get_string(rst_test))
    logResult(file_path_LSTM, get_string(rst_predict))


def logResult(file_path, text):

    fileIO.checkDirectory()

    f = open(file_path, 'a')

    f.write(text + "\n")
    f.close()


def get_string(array):

    rst = ''

    for i, val in enumerate(array):
        rst = rst + str(val[0]) + ','

    print("String rst: " + rst)

    return rst



def load_ANN(df_kp):
    file_paths = ['modelResults/20220610020403.hs']

    country_types = ['kp']

    data_types = ['us']

    load_ann_file(df_kp, country_types[0], data_types[0], file_paths[0])



def load_ann_file(df, market_type, data_type, file_path):
    # Normalize
    df = LSTM.normalize(df)

    df = df[LSTM.split:]

    feature_cols_us, feature_cols = loadData.get_feature_cols(market_type)
    label_cols = ['index_' + market_type]

    x_test = pd.DataFrame(df, columns=feature_cols_us).to_numpy()
    y_test = pd.DataFrame(df, columns=label_cols).to_numpy()

    model = tf.keras.models.load_model(file_path)
    model.summary()

    y_predict = model.predict(x_test)

    rst_test = []
    rst_predict = []

    for i in range(len(y_test)):
        rst_test.append(y_test[i])
        rst_predict.append(y_predict[i])

    logResult(file_path_ANN, file_path)

    logResult(file_path_ANN, get_string(rst_test))
    logResult(file_path_ANN, get_string(rst_predict))
