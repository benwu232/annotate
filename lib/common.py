__author__ = 'wb'

import os, sys
import urllib.request, urllib.error, urllib.parse
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
#from matplotlib.finance import candlestick
#import matplotlib.finance as mf
import matplotlib
import pylab
import talib
import json
from collections import OrderedDict
import glob
import datetime as dt
import re

from matplotlib.transforms import Bbox

matplotlib.rcParams.update({'font.size': 9})

data_dir = './'

def dt2str(dt, format="%Y-%m-%d_%H:%M:%S"):
    return dt.strftime(format)

def now2str(format="%Y-%m-%d_%H-%M-%S-%f"):
    #str_time = time.strftime("%Y-%b-%d-%H-%M-%S", time.localtime(time.time()))
    return dt.datetime.now().strftime(format)

def str2dt(time_str, format='%Y-%m-%d'):
    return dt.datetime.strptime(time_str, format)

def str2ts(date_str, time_str, format='%Y-%m-%dT%H:%M:%S'):
    dt_time = str2dt(date_str+'T'+time_str, format=format)
    return dt_time.timestamp()

def gen_file_list(root_dir, surfix=''):
    file_list = []
    match_name = os.path.join(root_dir, '*{}'.format(surfix))
    #tmp_file_list = glob.glob('%s*%s' % (root_dir, surfix))
    tmp_file_list = glob.glob(match_name)

    #remove directories
    file_list = [f for f in tmp_file_list if not os.path.isdir(f)]

    # process files recursively
    list_dirs = os.listdir(root_dir)
    for e in list_dirs:
        if e[0] == '.':
            continue

        abs_path = os.path.join(root_dir, e)
        if os.path.isdir(abs_path):
            file_list += gen_file_list(abs_path, surfix)

    return file_list

def gen_coarse_symbol_list(start='201408'):
    #zz = future_prefixes_2d[CZCE]
    now = dt.datetime.now()
    end_dt = now + dt.timedelta(days=365)
    time_dt = dt.datetime.strptime(start, "%Y%m")
    symbol_list = []
    while time_dt < end_dt:
        time_str = time_dt.strftime("%Y%m")
        for prefix in future_symbols_1d:
            symbol = prefix + time_str[2:6]
            symbol_list.append(symbol)
        time_dt += dt.timedelta(days=28)

    symbol_list = list(set(symbol_list))
    print('{} symbols generated.'.format(len(symbol_list)))
    return symbol_list

def get_symbol_files_dict(symbol_match=[], symbol_list=[], path_in='', start='201408', verbose=False):
    print('\nGet symbol files dictionary ...')
    file_list = gen_file_list(path_in, surfix='.csv')
    if symbol_list == []:
        symbol_list = gen_coarse_symbol_list(start=start)

    symbol_files_dict = OrderedDict()
    for k, symbol in enumerate(symbol_list):
        if verbose and k % 100 == 0:
            print(k, len(symbol_list), symbol)

        #check if the symbol is in category
        if symbol_match:
            in_category = False
            for category in symbol_match:
                #if category == symbol:
                if re.match(category, symbol):
                    in_category = True
                    break
            if not in_category:
                continue

        symbol_files = []
        for f in file_list:
            f_symbol = re.search('\w{1,2}\d{4}_', f).group()
            if symbol == f_symbol[:-1]:
                symbol_files.append(f)

        #remove those found files in file_list and to reduce future search
        for sf in symbol_files:
            file_list.remove(sf)

        if symbol_files:
            #symbol_files.sort()
            #using basename to avoid influnce of different directories, re to unify the name format
            symbol_files.sort(key=lambda x: re.sub('-', '', os.path.basename(x)))
            symbol_files_dict[symbol] = symbol_files
            if verbose:
                print('get_symbol_files_dict find {}'.format(symbol))

        #if len(symbol_files_dict) >= 15:
        #    break

    print('Total symbol dict size {}'.format(len(symbol_files_dict)))
    #save2json('symbol_files_dict.json', symbol_files_dict)
    #exit()
    return symbol_files_dict

def load_files(file_list, active_info=[], start='2014-08-1', end='', pre_tick_offset=10000, verbose=False):
    start_idx = -1
    last_break = False
    df_data_right = pd.DataFrame()
    df_data_left = pd.DataFrame()
    start_dt = dt.datetime.strptime(start, "%Y-%m-%d")
    if end == '':
        end_dt = dt.datetime.now()
    else:
        end_dt = dt.datetime.strptime(end, "%Y-%m-%d")
    if active_info == []:
        active_start = start_dt
        active_end = end_dt
    else:
        active_start = dt.datetime.strptime(active_info[1], "%Y%m%d")
        active_end = dt.datetime.strptime(active_info[-1], "%Y%m%d")
    start_dt = max(start_dt, active_start)
    end_dt = min(end_dt, active_end)

    if start_dt >= end_dt:
        return df_data_right

    #Load data from start to end
    for k, f in enumerate(file_list):
        base_name = os.path.basename(f)
        split = re.split('_', base_name)
        split[1] = re.sub('-', '', split[1])
        file_date_str = split[1][:8]
        file_date_dt = dt.datetime.strptime(file_date_str, "%Y%m%d")
        if file_date_dt < start_dt:
            start_idx = k
            continue
        elif file_date_dt > end_dt:
            break

        if verbose:
            print('Loading {} ...'.format(f))
        df_tmp = pd.read_csv(f, encoding='GB2312', dtype={col: np.float32 for col in ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'OpenInterest']})
        if len(df_tmp) == 0:
            last_break = True
            continue
        df_start = dt.datetime.fromtimestamp(df_tmp.loc[0]['Timestamp'])
        preset_data = np.zeros(len(df_tmp))
        if not ((df_start.hour == 9 or df_start.hour == 21) and df_start.minute < 10):
            preset_data[0] = 10000.0
        if last_break:
            preset_data[0] = 10000.0
            last_break = False
        df_tmp['Break'] = pd.Series(preset_data, index=df_tmp.index)
        #print(df_tmp['LastPrice'].min(), df_tmp['LastPrice'].max())
        '''
        mplt.plot(df_tmp['Timestamp'], df_tmp['LastPrice'])
        mplt.show()
        '''
        df_data_right = df_data_right.append(df_tmp, ignore_index=True)

    #Load pre_offset symbols
    if start_idx < 0:
        return None

    for k in range(start_idx, -1, -1):
        f = file_list[k]
        if verbose:
            print('Loading {} ...'.format(f))
        df_tmp = pd.read_csv(f, encoding='GB2312', dtype={col: np.float32 for col in ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'OpenInterest']})
        if len(df_tmp) == 0:
            if len(df_data_left) > 0:
                df_data_left.set_value(0, 'Break', 10000.0)
            continue

        df_start = dt.datetime.fromtimestamp(df_tmp.loc[0]['Timestamp'])
        preset_data = np.zeros(len(df_tmp))
        if not ((df_start.hour == 9 or df_start.hour == 21) and df_start.minute < 10):
            preset_data[0] = 10000.0
        df_tmp['Break'] = pd.Series(preset_data, index=df_tmp.index)
        df_data_left = df_tmp.append(df_data_left, ignore_index=True)

        if len(df_data_left) > pre_tick_offset:
            break
    df_data_left = df_data_left[-pre_tick_offset:]

    #Concatenate them
    df_data = df_data_left.append(df_data_right)

    return df_data

def avg_downsample(df_origin, avg_period=120):
    df_down_sample_factor = avg_period
    for col in df_origin:
        #cal mean
        avg_values = talib.SMA(np.asarray(df_origin[col].values, dtype=np.float64), avg_period)
        df_origin[col] = np.asarray(avg_values, dtype=np.float32)

    #slice
    df_origin = df_origin.loc[::df_down_sample_factor]
    return df_origin


def load_data(symbol_match=[], start='2014-01-01', verbose=True):
    with open('future_active_symbol_span_5.json') as data_file:
        active_file_info = json.load(data_file)

    start1 = start[:4] + start[5:7]
    symbol_files_dict = get_symbol_files_dict(symbol_match=symbol_match,
                                              symbol_list=active_file_info.keys(),
                                              path_in='./data',
                                              start=start1,
                                              verbose=verbose)
    data_dict = {}
    del_list = []
    for symbol in symbol_files_dict:
        if symbol not in active_file_info:
            del_list.append(symbol)
            continue

        print('Loading {}'.format(symbol))
        tmp_df = load_files(symbol_files_dict[symbol], active_info=active_file_info[symbol],
                            start=start,
                            end=dt.date.today().strftime('%Y-%m-%d'),
                            pre_tick_offset=20000,
                            verbose=verbose)

        if tmp_df.empty:
            del_list.append(symbol)
            continue

        active_start_time = active_file_info[symbol][1]
        active_start_ts = str2ts(active_start_time, '00:00:00', format='%Y%m%dT%H:%M:%S') * 2

        #tmp_df['Timestamp'] = (tmp_df['Timestamp']).astype(int)
        tmp_df['Timestamp'] = (tmp_df['Timestamp']*2).astype(int) #tick period is 0.5s, *2 to make timestamp an int
        tmp_df.set_index('Timestamp', inplace=True)
        df_idx = tmp_df.index

        if active_start_ts < df_idx[0] or active_start_ts > df_idx[-1]:
            print('Wrong: {}, {}, {}, will delete'.format(df_idx[0], active_start_ts, df_idx[-1]))
            del_list.append(symbol)
            continue

        tmp_df = avg_downsample(tmp_df)
        data_dict[symbol] = tmp_df

    if not data_dict:
        print('Empty data_df, quit.')
        #exit()

    return data_dict


