import numpy as np
import yfinance as yf
import pandas as pd
import csv
from pathlib import Path
from pandas_datareader import data as pdr
from datetime import datetime

pathForS500 = 'csv/snp500.csv'
pathForN100 = 'csv/nasdaq100.csv'
pathForDw = 'csv/dowjones.csv'
pathForVIX = 'csv/vix.csv'
pathForKp = 'csv/kospi.csv'
pathForNk = 'csv/nikkei.csv'
pathForHs = 'csv/hangseng.csv'
pathForKWD = 'csv/KWD.csv'
pathForJPY = 'csv/JPY.csv'
pathForCNY = 'csv/CNY.csv'
preprocessedData = 'csv/p_data.csv'

startDate = "2001-01-01"
endDate = "2022-05-31"


def downloadFinanceCSVs():
    yf.pdr_override()
    s500 = pdr.get_data_yahoo("^GSPC", start=startDate, end=endDate)
    n100 = pdr.get_data_yahoo("^NDX", start=startDate, end=endDate)
    dw = pdr.get_data_yahoo("^DJI", start=startDate, end=endDate)
    vix = pdr.get_data_yahoo("^VIX", start=startDate, end=endDate)

    kp = pdr.get_data_yahoo("^KS11", start=startDate, end=endDate)
    nk = pdr.get_data_yahoo("^N225", start=startDate, end=endDate)
    hs = pdr.get_data_yahoo("^HSI", start=startDate, end=endDate)

    kwd = pdr.get_data_yahoo("KWD=X", start=startDate, end=endDate)
    jpy = pdr.get_data_yahoo("JPY=X", start=startDate, end=endDate)
    cny = pdr.get_data_yahoo("CNY=X", start=startDate, end=endDate)

    filePathForS500 = Path(pathForS500)
    filePathForN100 = Path(pathForN100)
    filePathForDw = Path(pathForDw)
    filePathForVIX = Path(pathForVIX)

    filePathForKp = Path(pathForKp)
    filePathForNk = Path(pathForNk)
    filePathForHs = Path(pathForHs)

    filePathForKWD = Path(pathForKWD)
    filePathForJPY = Path(pathForJPY)
    filePathForCNY = Path(pathForCNY)

    filePathForS500.parent.mkdir(parents=True, exist_ok=True)
    filePathForN100.parent.mkdir(parents=True, exist_ok=True)
    filePathForDw.parent.mkdir(parents=True, exist_ok=True)
    filePathForVIX.parent.mkdir(parents=True, exist_ok=True)
    filePathForKp.parent.mkdir(parents=True, exist_ok=True)
    filePathForNk.parent.mkdir(parents=True, exist_ok=True)
    filePathForHs.parent.mkdir(parents=True, exist_ok=True)
    filePathForKWD.parent.mkdir(parents=True, exist_ok=True)
    filePathForJPY.parent.mkdir(parents=True, exist_ok=True)
    filePathForCNY.parent.mkdir(parents=True, exist_ok=True)

    s500.to_csv(filePathForS500)
    n100.to_csv(filePathForN100)
    dw.to_csv(filePathForDw)
    vix.to_csv(filePathForVIX)

    kp.to_csv(filePathForKp)
    nk.to_csv(filePathForNk)
    hs.to_csv(filePathForHs)

    kwd.to_csv(filePathForKWD)
    jpy.to_csv(filePathForJPY)
    cny.to_csv(filePathForCNY)


def processAndSaveCSVs():
    csv_s500 = pd.read_csv(pathForS500)
    csv_n100 = pd.read_csv(pathForN100)
    csv_dw = pd.read_csv(pathForDw)
    csv_vix = pd.read_csv(pathForVIX)

    csv_kp = pd.read_csv(pathForKp)
    csv_nk = pd.read_csv(pathForNk)
    csv_hs = pd.read_csv(pathForHs)

    csv_kwd = pd.read_csv(pathForKWD)
    csv_jpy = pd.read_csv(pathForJPY)
    csv_cny = pd.read_csv(pathForCNY)

    df = csv_s500.copy();

    # -- Change Columns' Names
    df.rename(columns={"Date": "date"}, inplace=True)
    df.rename(columns={"Open": "open_s100"}, inplace=True)
    df.rename(columns={"High": "high_s100"}, inplace=True)
    df.rename(columns={"Low": "low_s100"}, inplace=True)
    df.rename(columns={"Close": "close_s100"}, inplace=True)
    df.rename(columns={"Adj Close": "adj_close_s100"}, inplace=True)
    df.rename(columns={"Volume": "volume_s100"}, inplace=True)

    # Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')

    # -- Check Dataframe Length --
    n = len(df.index)

    # Add next business day
    for i in df.index:
        if (i + 1) < n:
            df.loc[i, 'next_date'] = df.loc[i + 1]['date']

    df = addCSV(df, csv_n100, 'n100')
    df = addCSV(df, csv_dw, 'dw')
    df = addCSV(df, csv_vix, 'vix')

    df = addCSV(df, csv_kwd, 'kwd')
    df = addCSV(df, csv_jpy, 'jpy')
    df = addCSV(df, csv_cny, 'cny')

    df = addCSV(df, csv_kp, 'kp')
    df = addCSV(df, csv_nk, 'nk')
    df = addCSV(df, csv_hs, 'hs')

    df = addNextDayCSV(df, csv_kp, 'kp')
    df = addNextDayCSV(df, csv_nk, 'nk')
    df = addNextDayCSV(df, csv_hs, 'hs')

    # # Add Korea, Japan, China Stock Info
    # for i in df.index:
    #
    #     kp_val = csv_kp.loc[csv_kp['Date'] == df.loc[i, 'next_date']]
    #     nk_val = csv_nk.loc[csv_nk['Date'] == df.loc[i, 'next_date']]
    #     hs_val = csv_hs.loc[csv_hs['Date'] == df.loc[i, 'next_date']]
    #
    #     if not kp_val.empty:
    #         df.loc[i, 'kp_price'] = kp_val['Open'].values[0]
    #
    #     if not nk_val.empty:
    #         df.loc[i, 'nk_price'] = nk_val['Open'].values[0]
    #
    #     if not hs_val.empty:
    #         df.loc[i, 'hs_price'] = hs_val['Open'].values[0]

    print(df)
    print(df.info())
    print(df.columns)

    # df = df.dropna(axis=0)
    # print(df.isna())

    saveProcessedCSV(df)


def saveProcessedCSV(df):
    df.to_csv(preprocessedData)
    print("--- Save Completed ---")


def loadProcessedCSV():
    df = pd.read_csv(preprocessedData)
    print("--- Load Completed ---")
    print("--- NaN Info ---")
    print(df.isna().sum())
    print("--- Null Info ---")
    print(df.isnull().sum())
    print("--- Statistics Info ---")
    print(df.describe())
    print("--- Columns Info ---")
    print(df.columns)
    return df


def getDay(date):
    temp = pd.Timestamp(date)
    return temp.dayofweek + 1


def addCSV(df1, df2, name):
    df1['open_' + name] = np.NaN
    df1['high_' + name] = np.NaN
    df1['low_' + name] = np.NaN
    df1['open_' + name] = np.NaN
    df1['close_' + name] = np.NaN
    df1['adj_close_' + name] = np.NaN
    df1['volume_' + name] = np.NaN

    for i in df1.index:

        row = df2.loc[df2['Date'] == df1.loc[i, 'date']]

        if not row.empty:
            df1.loc[i, 'open_' + name] = row['Open'].values[0]
            df1.loc[i, 'high_' + name] = row['High'].values[0]
            df1.loc[i, 'low_' + name] = row['Low'].values[0]
            df1.loc[i, 'close_' + name] = row['Open'].values[0]
            df1.loc[i, 'close_n100'] = row['Close'].values[0]
            df1.loc[i, 'adj_close_' + name] = row['Adj Close'].values[0]
            df1.loc[i, 'volume_' + name] = row['Volume'].values[0]

    return df1


def addNextDayCSV(df1, df2, name):
    df1['index_' + name] = np.NaN

    for i in df1.index:

        row = df2.loc[df2['Date'] == df1.loc[i, 'next_date']]

        if not row.empty:
            df1.loc[i, 'index_' + name] = row['Open'].values[0]

    return df1


def loadProcessedCSV():
    df = pd.read_csv(preprocessedData)
    print("--- Load Completed ---")
    print("--- NaN Info ---")
    print(df.isna().sum())
    print("--- Null Info ---")
    print(df.isnull().sum())
    print("--- Statistics Info ---")
    print(df.describe())
    print("--- Columns Info ---")
    print(df.columns)

    return df