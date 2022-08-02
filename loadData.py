import numpy as np
import yfinance as yf
import pandas as pd
import csv
from pathlib import Path
from pandas_datareader import data as pdr
from datetime import datetime

import readCSV

cols_kp = ['open_s100', 'high_s100', 'low_s100', 'close_s100',
              'adj_close_s100', 'volume_s100','open_n100', 'high_n100',
              'low_n100', 'close_n100', 'adj_close_n100', 'volume_n100', 'open_dw',
              'high_dw', 'low_dw', 'close_dw', 'adj_close_dw', 'volume_dw',
              'open_vix', 'high_vix', 'low_vix', 'close_vix', 'adj_close_vix',
              'open_kwd', 'high_kwd', 'low_kwd',
              'close_kwd', 'adj_close_kwd', 'open_kp',
              'high_kp', 'low_kp', 'close_kp', 'adj_close_kp', 'volume_kp', 'index_kp']

feature_cols_us_kp = ['open_s100', 'high_s100', 'low_s100', 'close_s100',
                      'adj_close_s100', 'volume_s100', 'open_n100', 'high_n100',
                      'low_n100', 'close_n100', 'adj_close_n100', 'volume_n100', 'open_dw',
                      'high_dw', 'low_dw', 'close_dw', 'adj_close_dw', 'volume_dw',
                      'open_vix', 'high_vix', 'low_vix', 'close_vix', 'adj_close_vix',
                      'open_kwd', 'high_kwd', 'low_kwd',
                      'close_kwd', 'adj_close_kwd', 'open_kp',
                      'high_kp', 'low_kp', 'close_kp', 'adj_close_kp', 'volume_kp']


feature_cols_kp = ['open_kp', 'high_kp', 'low_kp', 'close_kp', 'adj_close_kp', 'volume_kp']

cols_nk = ['open_s100', 'high_s100', 'low_s100', 'close_s100',
           'adj_close_s100', 'volume_s100', 'open_n100', 'high_n100',
           'low_n100', 'close_n100', 'adj_close_n100', 'volume_n100', 'open_dw',
           'high_dw', 'low_dw', 'close_dw', 'adj_close_dw', 'volume_dw',
           'open_vix', 'high_vix', 'low_vix', 'close_vix', 'adj_close_vix',
           'open_jpy', 'high_jpy', 'low_jpy',
           'close_jpy', 'adj_close_jpy', 'open_nk',
           'high_nk', 'low_nk', 'close_nk', 'adj_close_nk', 'volume_nk', 'index_nk']

feature_cols_us_nk = ['open_s100', 'high_s100', 'low_s100', 'close_s100',
                      'adj_close_s100', 'volume_s100', 'open_n100', 'high_n100',
                      'low_n100', 'close_n100', 'adj_close_n100', 'volume_n100', 'open_dw',
                      'high_dw', 'low_dw', 'close_dw', 'adj_close_dw', 'volume_dw',
                      'open_vix', 'high_vix', 'low_vix', 'close_vix', 'adj_close_vix',
                      'open_jpy', 'high_jpy', 'low_jpy',
                      'close_jpy', 'adj_close_jpy', 'open_nk',
                      'high_nk', 'low_nk', 'close_nk', 'adj_close_nk', 'volume_nk']

feature_cols_nk = ['open_nk', 'high_nk', 'low_nk', 'close_nk', 'adj_close_nk', 'volume_nk']

cols_hs = ['open_s100', 'high_s100', 'low_s100', 'close_s100',
           'adj_close_s100', 'volume_s100', 'open_n100', 'high_n100',
           'low_n100', 'close_n100', 'adj_close_n100', 'volume_n100', 'open_dw',
           'high_dw', 'low_dw', 'close_dw', 'adj_close_dw', 'volume_dw',
           'open_vix', 'high_vix', 'low_vix', 'close_vix', 'adj_close_vix',
           'open_cny', 'high_cny', 'low_cny',
           'close_cny', 'adj_close_cny', 'open_hs',
           'high_hs', 'low_hs', 'close_hs', 'adj_close_hs', 'volume_hs', 'index_hs']

feature_cols_us_hs = ['open_s100', 'high_s100', 'low_s100', 'close_s100',
                      'adj_close_s100', 'volume_s100', 'open_n100', 'high_n100',
                      'low_n100', 'close_n100', 'adj_close_n100', 'volume_n100', 'open_dw',
                      'high_dw', 'low_dw', 'close_dw', 'adj_close_dw', 'volume_dw',
                      'open_vix', 'high_vix', 'low_vix', 'close_vix', 'adj_close_vix',
                      'open_cny', 'high_cny', 'low_cny',
                      'close_cny', 'adj_close_cny', 'open_hs',
                      'high_hs', 'low_hs', 'close_hs', 'adj_close_hs', 'volume_hs']

feature_cols_hs = ['open_hs', 'high_hs', 'low_hs', 'close_hs', 'adj_close_hs', 'volume_hs']


def load_dataframe(type):

    df = pd.read_csv(readCSV.preprocessedData)
    cols = ''

    if type == 'kp':
        cols = cols_kp

    elif type == 'nk':
        cols = cols_nk

    elif type == 'hs':
        cols = cols_hs

    df = pd.DataFrame(df, columns=cols)
    df = df.replace(0, np.NaN)

    df = df.dropna(axis=0)

    return df


def get_feature_cols(type):
    if type == 'kp':
        return feature_cols_us_kp, feature_cols_kp

    elif type == 'nk':
        return feature_cols_us_nk, feature_cols_nk

    return feature_cols_us_hs, feature_cols_hs
