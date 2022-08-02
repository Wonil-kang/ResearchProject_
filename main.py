# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import yfinance as yf
import pandas as pd

import DT
import LSTM
import LR
import NN
import RF
import RNN
import SVM
import csv
from pathlib import Path
from pandas_datareader import data as pdr

import fileIO
import loadData
import load_result
from readCSV import downloadFinanceCSVs, processAndSaveCSVs, loadProcessedCSV


def main():

    ## -------------------- PREPROCESSING -------------------- ##
    # downloadFinanceCSVs()
    # processAndSaveCSVs()

    ## ---------------------- LOAD DATA ---------------------- ##

    df_kp = loadData.load_dataframe('kp')
    df_nk = loadData.load_dataframe('nk')
    df_hs = loadData.load_dataframe('hs')

    # print(df_kp)

    ## ------------------- RUN  LSTM ------------------- ##

    # LSTM.run_LSTM(df_kp, 'kp')
    # LSTM.run_LSTM(df_nk, 'nk')
    # LSTM.run_LSTM(df_hs, 'hs')

    ## ------------------- RUN  RANDOM FOREST ------------------- ##

    # RF.run_RF(df_kp, 'kp')
    # RF.run_RF(df_nk, 'nk')
    # RF.run_RF(df_hs, 'hs')

    ## ------------------- RUN  LINEAR REGRESSION ------------------- ##

    LR.run_LR(df_kp, 'kp')
    LR.run_LR(df_nk, 'nk')
    LR.run_LR(df_hs, 'hs')

    ## ------------------- RUN  RECULSIVE NEURAL NETWORK ------------------- ##

    # RNN.run_RNN(df_kp, 'kp')
    # RNN.run_RNN(df_nk, 'nk')
    # RNN.run_RNN(df_hs, 'hs')

    ## ------------------- RUN  ALTIFIAL NEURAL NETWORK ------------------- ##

    # NN.run_NN(df_kp, 'kp')
    # NN.run_NN(df_nk, 'nk')
    # NN.run_NN(df_hs, 'hs')

    ## ------------------- RUN  SUPPORT VECTOR MACHINE ------------------- ##
    #
    # SVM.run_SVM(df_kp, 'kp')
    # SVM.run_SVM(df_nk, 'nk')
    # SVM.run_SVM(df_hs, 'hs')

    # load_result.load_LSTM(df_kp, df_nk, df_hs)

    # load_result.load_ANN(df_kp)

    # NN.run(df)

if __name__ == "__main__":
    main()
