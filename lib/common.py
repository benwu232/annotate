__author__ = 'wb'

import os, sys
import urllib.request, urllib.error, urllib.parse
import time
import datetime
import pickle
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

color_dict = {
    0: 'gray',
    1: 'r', #buy
    2: 'g', #sell
    3: '#ff028d', #short
    4: '#9a0eea', #cover
    5: 'orange', #up trend
    6: '#069af3', #down trend
    7: 'w' #to be define
}

def label2color(label_line, label_colors, start, end):
    for k in range(start, end+1):
        label_colors[k] = color_dict[label_line[k]]

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

def save_dump(dump_data, out_file):
    with open(out_file, 'wb') as fp:
        print('Writing to %s.' % out_file)
        #pickle.dump(dump_data, fp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dump_data, fp)

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

def load_files(data_dir, file_list, active_info=[], start='2014-08-1', end='', pre_offset=10000, verbose=False):
    start_idx = active_info[1]
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
        active_start = os.path.basename(file_list[active_info[1]]).split('_')[-1][:8]
        active_start = dt.datetime.strptime(active_start, "%Y%m%d")
        active_end = os.path.basename(file_list[active_info[-1]]).split('_')[-1][:8]
        active_end = dt.datetime.strptime(active_end, "%Y%m%d")
    start_dt = max(start_dt, active_start)
    end_dt = min(end_dt, active_end)

    if start_dt >= end_dt:
        return df_data_right

    #change file path


    #Load data from start to end
    for k in range(active_info[1], active_info[-1]):
        base_name = os.path.basename(file_list[k])
        file_date_str = base_name.split('_')[-1][:8]
        file_date_dt = dt.datetime.strptime(file_date_str, "%Y%m%d")
        if file_date_dt < start_dt:
            start_idx = k
            continue
        elif file_date_dt > end_dt:
            break
        real_file_path = os.path.join(data_dir, base_name)

        if verbose:
            print('Loading {} ...'.format(real_file_path))
        df_tmp = pd.read_csv(real_file_path, encoding='GB2312', dtype={col: np.float32 for col in ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'OpenInterest']})
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
    #if start_idx < 0:
    #    return None

    for k in range(start_idx-1, -1, -1):
        base_name = os.path.basename(file_list[k])
        real_file_path = os.path.join(data_dir, base_name)
        if verbose:
            print('Loading {} ...'.format(real_file_path))
        df_tmp = pd.read_csv(real_file_path, encoding='GB2312', dtype={col: np.float32 for col in ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'OpenInterest']})
        if len(df_tmp) == 0:
            if len(df_data_left) > 0:
                df_data_left.at[0, 'Break'] = 10000.0
            continue

        df_start = dt.datetime.fromtimestamp(df_tmp.loc[0]['Timestamp'])
        preset_data = np.zeros(len(df_tmp))
        if not ((df_start.hour == 9 or df_start.hour == 21) and df_start.minute < 10):
            preset_data[0] = 10000.0
        df_tmp['Break'] = pd.Series(preset_data, index=df_tmp.index)
        df_data_left = df_tmp.append(df_data_left, ignore_index=True)

        if len(df_data_left) > pre_offset:
            break
    df_data_left = df_data_left[-pre_offset:]

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

def sel_active_futures(futures_dict, active_distance_rate_min=1.2, active_variation_min=8000,
                       active_span_size=30, day_pre_offset=20, avg_factor=5):
    del_list = []
    momentum = 1 - 1/avg_factor
    print('Analyzing and selecting active futures ...')
    for symbol in futures_dict:
        variation_num_avg = 0
        activity_rate_avg = 0
        active_start = -1
        active_end = -1
        for k, (symbol_file, variation_num, activity_rate) in enumerate(zip(futures_dict[symbol]['files'], futures_dict[symbol]['variation'], futures_dict[symbol]['active_distance_rate'])):
            variation_num_avg = variation_num_avg * momentum + variation_num * (1 - momentum)
            activity_rate_avg = activity_rate_avg * momentum + activity_rate * (1 - momentum)
            if variation_num_avg >= active_variation_min and activity_rate_avg >= active_distance_rate_min:
                if active_start < 0:
                    active_start = k - avg_factor
            else:
                if active_end < 0 and active_start >= 0:
                    active_end = k - avg_factor
                    break

        #if no active span or active span is too short
        if active_start < 0 or active_end < 0 or active_end - active_start < active_span_size:
            del_list.append(symbol)
        else:
            pre = max(0, active_start - day_pre_offset)
            futures_dict[symbol]['active_span'] = [pre, active_start, active_end]

    #delete bad symbols
    for symbol in del_list:
        futures_dict.pop(symbol, None)

    return futures_dict


def load_futures_data(data_dir, futures_dict, symbol_match=None, start='2014-01-01', end='', pre_offset=4096, verbose=True):
    data_dict = {}
    del_list = []
    if end == '':
        end=dt.date.today().strftime('%Y-%m-%d'),

    symbol_list = []
    if type(symbol_match) is str:
        for sym in futures_dict.keys():
            if symbol_match in sym:
                symbol_list.append(sym)

    for symbol in symbol_list:
        print('Loading {}'.format(symbol))
        tmp_df = load_files(data_dir,
                            futures_dict[symbol]['files'],
                            active_info=futures_dict[symbol]['active_span'],
                            start=start,
                            end=end,
                            pre_offset=pre_offset,
                            verbose=verbose)

        if tmp_df.empty:
            del_list.append(symbol)
            continue

        active_start_time = futures_dict[symbol]['files'][futures_dict[symbol]['active_span'][1]]
        active_start_time = active_start_time.split('_')[-1][:8]
        active_start_ts = str2ts(active_start_time, '00:00:00', format='%Y%m%dT%H:%M:%S')

        tmp_df['Timestamp'] = (tmp_df['Timestamp']).astype(int)
        tmp_df.set_index('Timestamp', inplace=True)
        df_idx = tmp_df.index

        if active_start_ts < df_idx[0] or active_start_ts > df_idx[-1]:
            print('Wrong: {}, {}, {}, will delete'.format(df_idx[0], active_start_ts, df_idx[-1]))
            del_list.append(symbol)
            continue

        data_dict[symbol] = tmp_df

    if not data_dict:
        print('Empty data_df, quit.')
        #exit()

    return data_dict


def datapro(data_dict):
    for symbol in data_dict:
        price_line = data_dict[symbol]['LastPrice'].values
        volume_line = data_dict[symbol]['Volume'].values
        for k in range(len(price_line)-2, -1, -1):
            if np.isnan(price_line[k]):
                price_line[k] = price_line[k+1]
            if np.isnan(volume_line[k]):
                volume_line[k] = volume_line[k+1]
        data_dict[symbol]['LastPrice'] = price_line
        data_dict[symbol]['Volume'] = volume_line
    return data_dict
