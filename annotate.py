__author__ = 'wb'

import urllib.request, urllib.error, urllib.parse
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
#from matplotlib.finance import candlestick
#import matplotlib.finance as mf
import matplotlib
import pylab
import talib
import sys

from matplotlib.transforms import Bbox

from lib.common import *

matplotlib.rcParams.update({'font.size': 9})


RIGHT_CORNER = 0.93
LEFT_CORNER = 0.01
class Visual:
    def __init__(self, data_pool, mydir, before=60, after=10, ma1=20, ma2=40, ma3=60, ma4=80, ma5=100, ma6=120, zoom_ratio = 1.2):
        self.data_pool = data_pool
        self.mydir = mydir
        self.before = before
        self.after = after
        self.ma1 = ma1
        self.ma2 = ma2
        self.ma3 = ma3
        self.ma4 = ma4
        self.ma5 = ma5
        self.ma6 = ma6
        self.zoom_ratio = zoom_ratio
        self.key_footnote = RIGHT_CORNER
        self.mouse_footnote = LEFT_CORNER
        self.plt_show = 0
        self.run_idx = 0

        #self.calculated = {}
        self.candle_stick_data = {}
        self.date2num_list = {}

        self.av1 = {}
        self.av2 = {}
        self.av3 = {}
        self.av4 = {}
        self.av5 = {}
        self.av6 = {}

        self.ma_v = {}

        self.sma5 = {}

        self.av_s = {}
        self.av_l = {}
        self.av_m = {}
        self.av_glue_idx = {}
        self.av_s1 = {}
        self.av_l1 = {}
        self.av_m1 = {}
        self.av_glue_idx1 = {}
        self.high_values = {}
        self.low_values = {}
        self.rsv = {}
        self.volume_glue_idx = {}
        self.ma_obv1 = {}
        self.ma_obv = {}
        self.nobv = {}

        self.ma_obv20 = {}
        self.ma_obv90 = {}
        self.ma_obv60 = {}
        self.obv_glue = {}

        self.efficiency = {}
        self.alr_slope_v = {}
        self.alr_slope_p = {}

        self.upperband = {}
        self.middleband = {}
        self.lowerband = {}

        self.slowk = {}
        self.slowd = {}
        self.slowj = {}

        self.dif = {}
        self.dem = {}
        self.macdhist = {}

        self.pos_list = []
        self.neg_list = []

        self.trade_record = []
        self.cur_time = ''

        self.hide_center_day = 0
        self.hide_circle = 1
        self.hide_time = 0
        self.mark_list = [[] for k in range(10)]
        self.undo_keys = []

        self.cal_tickers = {}

        self.predict_values = {}
        self.slope_list = {}
        self.pr_sl = {}

        self.left_time = '0000-00-00'
        self.right_time = '0000-00-00'
        self.htf_list = []


    def __cal_win__(self):
        self.win_start = self.center_idx - self.before
        if self.win_start < 0:
            self.win_start = 0
        self.win_end = self.win_start + self.before + self.after
        if self.win_end >= len(self.data_df.data):
            self.win_end = len(self.data_df.data) - 1

    def __step_center__(self, step):
        self.center_idx += step
        win_size = self.win_end - self.win_start + 1

        self.before += step
        if self.before > win_size:
            self.before = win_size
        elif self.before < 0:
            self.before = 0

        self.after -= step
        if self.after > win_size:
            self.after = win_size
        elif self.after < 0:
            self.after = 0

        if self.center_idx < 0:
            self.center_idx = 0
        elif self.center_idx >= len(self.data_df.data):
            self.center_idx = len(self.data_df.data) - 1

        if self.center_idx < self.win_start:
            self.win_end -= self.win_start - self.center_idx
            self.win_start = self.center_idx

        if self.center_idx > self.win_end:
            self.win_start += self.center_idx - self.win_end
            self.win_end = self.center_idx

    def __step_win__(self, step):
        self.center_idx += step
        self.win_end += step
        self.win_start += step

        if self.center_idx < 0:
            self.center_idx = 0
        elif self.center_idx >= len(self.data_df.data):
            self.center_idx = len(self.data_df.data) - 1

        if self.win_start < 0:
            self.win_start = 0
        elif self.win_start >= len(self.data_df.data):
            self.win_start = len(self.data_df.data) - 1

        if self.win_end < 0:
            self.win_end = 0
        elif self.win_end >= len(self.data_df.data):
            self.win_end = len(self.data_df.data) - 1

    def cal_all_data(self, ticker):
        #check if the key has already been calculated
        if ticker in self.cal_tickers:
            return
        else:
            self.cal_tickers[ticker] = 1


        #Calculate indicators
        x = 0
        y = len(self.data_df)

        # the data has already been calculated
        #if ticker in self.calculated:
        #    return

        self.candle_stick_data[ticker] = []
        self.date2num_list[ticker] = []
        while x < y:
            date_num = pd.to_datetime(self.data_df.index/2, unit='s')
            appendLine = date_num, self.data_df.data['Open'].iloc[x], self.data_df.data['Close'].iloc[x], self.data_df.data['High'].iloc[x], self.data_df.data['Low'].iloc[x], self.data_df.data['Volume'].iloc[x]
            self.candle_stick_data[ticker].append(appendLine)
            self.date2num_list[ticker].append(date_num)
            x+=1

        self.av1[ticker] = talib.SMA(self.data_df.data['Close'].values, self.ma1)
        self.av2[ticker] = talib.SMA(self.data_df.data['Close'].values, self.ma2)
        self.av3[ticker] = talib.SMA(self.data_df.data['Close'].values, self.ma3)

        self.av4[ticker] = talib.SMA(self.data_df.data['Close'].values, self.ma4)
        self.av5[ticker] = talib.SMA(self.data_df.data['Close'].values, self.ma5)
        self.av6[ticker] = talib.SMA(self.data_df.data['Close'].values, self.ma6)

        self.ma_v[ticker] = talib.SMA(self.data_df.data['Volume'].values, 10)

        self.sma5[ticker] = talib.SMA(self.data_df.data['Close'].values, 5)

        #self.av4[ticker] = talib.KAMA(self.stock.data['Close'].values, 20)
        #self.av5[ticker], b = talib.MAMA(self.stock.data['Close'].values)
        #self.av6[ticker] = talib.MAVP(self.stock.data['Close'].values)

        self.av_s[ticker] = talib.SMA(self.data_df.data['Close'].values, 20)
        self.av_l[ticker] = talib.SMA(self.data_df.data['Close'].values, 90)
        self.av_m[ticker] = talib.SMA(self.data_df.data['Close'].values, 60)
        self.av_glue_idx[ticker] = (self.av_s[ticker] - self.av_l[ticker]) / self.av_m[ticker]

        self.av_s1[ticker] = self.av_s[ticker]
        self.av_l1[ticker] = self.av_m[ticker]
        self.av_m1[ticker] = talib.SMA(self.data_df.data['Close'].values, 40)
        self.av_glue_idx1[ticker] = (self.av_s1[ticker] - self.av_l1[ticker]) / self.av_m1[ticker]

        #self.ma_obv[ticker] = talib.SMA(self.obv[ticker], 5)
        #self.volume_glue_idx[ticker] = talib.OBV(self.stock.data['Close'].values, self.stock.data['Volume'].values)
        self.ma_obv[ticker] = qute_algo.cal_ma_obv(self.data_df.data['Close'].values, self.data_df.data['Volume'].values, 40)
        self.ma_obv1[ticker] = qute_algo.cal_ma_obv1(self.data_df.data, 40)
        self.nobv[ticker] = qute_algo.cal_nobv(self.data_df.data['Close'].values, self.data_df.data['Volume'].values, 40, 40)
        #obv_single = misc.cal_ma_obv_single(self.stock.data)
        #self.ma_obv[ticker], dummy = talib.MAMA(obv_single)
        '''
        self.ma_obv20[ticker] = misc.cal_ma_obv(self.stock.data['Close'].values, self.stock.data['Volume'].values, 20)
        self.ma_obv90[ticker] = misc.cal_ma_obv(self.stock.data['Close'].values, self.stock.data['Volume'].values, 90)
        self.ma_obv60[ticker] = misc.cal_ma_obv(self.stock.data['Close'].values, self.stock.data['Volume'].values, 60)
        self.obv_glue[ticker] = (self.ma_obv20[ticker] - self.ma_obv90[ticker]) / self.ma_obv60[ticker]
        '''


        #self.volume_glue_idx[ticker] = misc.cal_volume_glue_idx(self.stock.data['Close'].values, self.stock.data['Volume'].values, short=20, long=90, middle=60)


        self.upperband[ticker], self.middleband[ticker], self.lowerband[ticker] = talib.BBANDS(self.data_df.data['Close'].values, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

        self.slowk[ticker], self.slowd[ticker] = talib.STOCH(self.data_df.data['High'].values, self.data_df.data['Low'].values, self.data_df.data['Close'].values)
        self.slowj[ticker] = 3*self.slowk[ticker] - 2*self.slowd[ticker]

        self.dif[ticker], self.dem[ticker], self.macdhist[ticker] = talib.MACD(self.data_df.data['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)


        #cal RSV
        length = len(self.data_pool[ticker].data['Close'])
        self.rsv_period = 240
        self.high_values[ticker] = [0 for k in range(0, length)]
        self.low_values[ticker] = [0 for k in range(0, length)]
        self.rsv[ticker] = [0 for k in range(0, length)]

        k = 0
        while k < length:
            #if k == 0 or k == 1:
            if k == 0:
                self.high_values[ticker][k] = (self.data_pool[ticker].data['Close'].iloc[k])
                self.low_values[ticker][k] = (self.data_pool[ticker].data['Close'].iloc[k])
            elif k < self.rsv_period:
                self.high_values[ticker][k] = (max(self.data_pool[ticker].data['Close'].iloc[:k]))
                self.low_values[ticker][k] = (min(self.data_pool[ticker].data['Close'].iloc[:k]))
            else:
                self.high_values[ticker][k] = (max(self.data_pool[ticker].data['Close'].iloc[k - self.rsv_period:k]))
                self.low_values[ticker][k] = (min(self.data_pool[ticker].data['Close'].iloc[k - self.rsv_period:k]))

            rsv = (self.data_pool[ticker].data['Close'].iloc[k] - self.low_values[ticker][k]) / (self.high_values[ticker][k] - self.low_values[ticker][k])
            self.rsv[ticker][k] = (rsv)
            k+=1

        #cal efficiency
        '''
        self.efficiency[ticker] = np.zeros(length)
        idx = 0
        while idx < length:
            self.efficiency[ticker][idx] = misc.price_trace_efficiency(self.stock_pool[ticker], idx, 20)
            idx += 1

        '''


        '''
        self.alr_slope_v[ticker] = np.zeros(length)
        self.alr_slope_p[ticker] = np.zeros(length)
        idx = 240
        while idx < length:
            alr_par_v, alr_slope_v, alr_std_v, alr_period_v = misc.adapt_lr1(self.ma_obv1[ticker], idx, alr_start=30, max_alr_period=240)
            self.alr_slope_v[ticker][idx] = alr_slope_v
            alr_par_p, alr_slope_p, alr_std_p, alr_period_p = misc.adapt_lr1(self.stock_pool[ticker].data['Close'].values, idx, alr_start=30, max_alr_period=240)
            self.alr_slope_p[ticker][idx] = alr_slope_p
            idx += 1
        '''


        '''
        #try new strategy
        f = open('tmp.log', 'w')
        self.predict_values[ticker] = np.zeros(length)
        self.slope_list[ticker] = np.zeros(length)
        self.pr_sl[ticker] = np.zeros(length)
        idx = 240
        last_predict_deviation = 0.0
        deviation = 0.0
        rise = 0
        while idx < length:

            middle, m_slope, std_min, std_min_n = misc.linear_regression(self.stock, idx-1)
            mid_a = middle[0]
            mid_b = middle[1]
            self.slope_list[ticker][idx] = mid_a

            #predict_y = mid_a * idx + mid_b + last_predict_deviation
            predict_y = mid_a * idx + mid_b
            deviation = (self.stock.data['Close'].iloc[idx] - predict_y) / self.stock.data['Close'].iloc[idx]
            last_predict_deviation = deviation
            self.predict_values[ticker][idx] = deviation

            if self.stock.data['Close'].iloc[idx] > self.stock.data['Close'].iloc[idx-1]:
                rise = 1
            else:
                rise = 0

            f.write('%s, std_min = %f, std_min_n = %d, slope = %f, deviation = %f, %f\n' % (self.stock.data['Date'].iloc[idx], std_min, std_min_n, m_slope, deviation, self.stock.data['Close'].iloc[idx]))

            idx += 1

        self.pr_sl[ticker] = self.predict_values[ticker] + self.slope_list[ticker]
        '''



    def draw_circle(self, ticker, x, max_circle_radius=500):
        #Close > Open
        if self.candle_stick_data[ticker][x][2] > self.candle_stick_data[ticker][x][1]:
            circle_color = 'r'
            circle_alpha = 0.2
        else:
            circle_color = '#00ffe8'
            circle_alpha = 0.1

        max_win_volume = 0.0
        for e in self.candle_stick_data[ticker][self.win_start:self.win_end]:
            if max_win_volume < e[-1]:
                max_win_volume = e[-1]
        radius = np.sqrt(self.candle_stick_data[ticker][x][-1]) * max_circle_radius / np.sqrt(max_win_volume)

        return radius, circle_color, circle_alpha


    def draw(self, ticker, action):
        center_idx_color = 'm'
        if action == -1:
            center_idx_color = 'g'


        self.ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, axisbg='#07000d')

        if not self.hide_circle:
            win_idx = self.win_start
            while win_idx < self.win_end:
                radius, circle_color, circle_alpha = self.draw_circle(ticker, win_idx)
                self.ax1.plot(self.date2num_list[ticker][win_idx], self.candle_stick_data[ticker][win_idx][2], '.', color=circle_color, markersize=radius, alpha=circle_alpha)
                win_idx += 1

        mf.candlestick(self.ax1, self.candle_stick_data[ticker][self.win_start:self.win_end+1], width=0.6, colorup='r', colordown='#00ffe8', alpha=0.85)


        # Draw vertical line to indicate the center day
        if not self.hide_center_day:
            self.ax1.axvline(self.date2num_list[ticker][self.center_idx], ymax=(self.ax1.axis())[-1], color = center_idx_color, alpha=0.5)
            radius, circle_color, circle_alpha = self.draw_circle(ticker, self.center_idx)
            #print self.center_idx
            self.ax1.plot(self.date2num_list[ticker][self.center_idx], self.candle_stick_data[ticker][self.center_idx][2], '.', color=circle_color, markersize=radius, alpha=0.3)

        #Draw the shadow line, candlestick doesn't have this parameter
        self.ax1.plot([self.date2num_list[ticker][self.win_start:self.win_end+1],self.date2num_list[ticker][self.win_start:self.win_end+1]], [self.data_df.data['Low'][self.win_start:self.win_end + 1].values, self.data_df.data['High'][self.win_start:self.win_end + 1].values], 'w')
        #self.ax1.axvline(date_num_list[-SP:], ymin=stock_cal['Close'][-SP:].values, ymax=stock_cal['High'][-SP:].values, color = 'w')

        Label1 = str(self.ma1)+' SMA'
        Label2 = str(self.ma2)+' SMA'
        Label3 = str(self.ma3)+' SMA'
        Label4 = str(self.ma4)+' SMA'
        Label5 = str(self.ma5)+' SMA'
        Label6 = str(self.ma6)+' SMA'

        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.av1[ticker][self.win_start:self.win_end+1],'#ff8000',label=Label1, linewidth=1)
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.av2[ticker][self.win_start:self.win_end+1],'y',label=Label2, linewidth=1)
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.av3[ticker][self.win_start:self.win_end+1],'g',label=Label3, linewidth=1)
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.av4[ticker][self.win_start:self.win_end+1],'m',label=Label4, linewidth=1)
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.av5[ticker][self.win_start:self.win_end+1],'#0000aa',label=Label5, linewidth=1)
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.av6[ticker][self.win_start:self.win_end+1],'#00cccc',label=Label6, linewidth=1)

        # Bollinger
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.upperband[ticker][self.win_start:self.win_end+1],'c', linewidth=1, linestyle = '-.')
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.middleband[ticker][self.win_start:self.win_end+1],'c', linewidth=1, linestyle = '-')
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.win_end+1],self.lowerband[ticker][self.win_start:self.win_end+1],'c', linewidth=1, linestyle = '-.')


        '''
        #####
        lowess_slope_list = []
        slope_list = []
        idx = self.win_start
        while idx <= self.center_idx:
            lowess, lowss_slope = misc.cal_lowess(self.stock, idx, 240)
            lowess_slope_list.append(lowess_slope*240)
            r, slope = misc.strategy(self.stock, idx)
            slope_list.append(slope*240)
            idx += 1
        '''

        #try new strategy
        '''
        regression, slope_240, peaks_max_group, peaks_min_group, peaks_max, peaks_min = misc.strategy3(self.stock, self.center_idx, self.center_idx+1-self.win_start)
        ava = talib.SMA(self.stock.data['Close'].values, 5)
        for p in peaks_max_group[-1]['data']:
            self.ax1.plot(self.date2num_list[ticker][p], ava[p], 'ro')
        for p in peaks_min_group[-1]['data']:
            self.ax1.plot(self.date2num_list[ticker][p], ava[p], 'go')

        diff = self.sma5[ticker][self.center_idx] - (regression[0] * self.center_idx + regression[1])

        print self.center_idx, self.stock.data['Date'].iloc[self.center_idx], slope_240, diff, diff/self.sma5[ticker][self.center_idx]
        '''


        '''
        points_top, pars_top, points_bottom, pars_bottom = misc.strategy2(self.stock, self.center_idx)
        p_idx = 0
        x_start = points_top[p_idx][-1]
        x_end = points_top[p_idx][0]

        while p_idx < len(points_top):
            if points_top[p_idx][0] <= self.win_start or points_top[p_idx][-1] >= self.win_end:
                break
            elif points_top[p_idx][0] < self.win_end:
                x_end = self.win_end
            elif points_top[p_idx][-1] > self.win_start:
                x_start = self.win_start

            x_arr = np.array(range(x_start, x_end+1))
            y_arr = pars_top[p_idx][0] * x_arr + pars_top[p_idx][1]
            y_arr[y_arr > max_high] = max_high
            y_arr[y_arr < min_low] = min_low
            self.ax1.plot(self.date2num_list[ticker][x_start:x_end+1], y_arr, 'w--')
            p_idx+=1
        '''



        #display the tagged points
        tagged_idx = 0
        tagged_len = len(self.tagged_points[ticker]['time'])
        while tagged_idx < tagged_len:
            time_idx = self.data_pool[ticker].time_str2idx(self.tagged_points[ticker]['time'][tagged_idx])
            if time_idx < self.win_start:
                tagged_idx += 1
                continue

            if time_idx > self.win_end:
                break

            # tag the chart with the values
            date_num = mdates.date2num(dt.datetime.strptime(self.tagged_points[ticker]['time'][tagged_idx], '%Y-%m-%d'))
            if self.tagged_points[ticker]['action'][tagged_idx] == '1':
                self.ax1.plot(date_num, self.middleband[ticker][time_idx], 'mo')
                self.ax1.axvline(date_num, ymax=(self.ax1.axis())[-1], color = 'm', alpha=0.5)
            elif self.tagged_points[ticker]['action'][tagged_idx] == '-1':
                self.ax1.plot(date_num, self.middleband[ticker][time_idx], 'go')
                self.ax1.axvline(date_num, ymax=(self.ax1.axis())[-1], color = 'g', alpha=0.5)
                #self.ax1.plot(date_num, self.stock_pool[ticker].data['Close'].iloc[time_idx], 'go')
            tagged_idx += 1

        '''
        #Display current index of prices and volumes
        self.ax1.text(self.key_footnote, 0.98,  '%s\n%s\n%s\n%s\n%s\n%sM\n%s\n%sM' %(self.stock.data['Date'].values[self.center_idx], str(self.stock.data['Open'].values[self.center_idx])[:5], str(self.stock.data['High'].values[self.center_idx])[:5], str(self.stock.data['Low'].values[self.center_idx])[:5], str(self.stock.data['Close'].values[self.center_idx])[:5], str(round(self.stock.data['Volume'].values[self.center_idx]/1000000.0)), str(self.win_end-self.win_start+1), str(round(self.ma_obv[ticker][self.center_idx]/1000000.0))),
        fontsize=10, alpha=1.0,
        horizontalalignment='left',
        verticalalignment='top',
        transform=self.ax1.transAxes, color=center_idx_color)
        '''

        #deceration
        self.ax1.grid(True, color='w')
        self.ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax1.yaxis.label.set_color("w")
        self.ax1.spines['bottom'].set_color("#5998ff")
        self.ax1.spines['top'].set_color("#5998ff")
        self.ax1.spines['left'].set_color("#5998ff")
        self.ax1.spines['right'].set_color("#5998ff")
        self.ax1.tick_params(axis='y', colors='w')
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
        self.ax1.tick_params(axis='x', colors='w')
        plt.ylabel('Price and Volume')

        maLeg = plt.legend(loc=9, ncol=2, prop={'size':7}, fancybox=True, borderaxespad=0.)
        maLeg.get_frame().set_alpha(0.4)
        textEd = pylab.gca().get_legend().get_texts()
        pylab.setp(textEd[0:5], color = 'w')

        #####Volume
        ax1v = self.ax1.twinx()
        #ax1v.fill_between(date_num_list[-SP:],volumeMin, stock_cal['Volume'][-SP:].values, facecolor='#00ffe8', alpha=.4)
        #ax1v.bar(date_num_list[-SP:], stock_cal['Volume'][-SP:].values, color='#00ffe8')
        #lfunc = lambda k: if stock_cal['Open'][k].values

        # Display the volume given by up or down
        volume_width = 0.6
        k = self.win_start
        while k <= self.win_end:
        #for k in range(SP):
            if self.data_df.data['Open'].values[k] < self.data_df.data['Close'].values[k]:
                c = 'r'
            else:
                c = '#00ffe8'
            ax1v.bar(self.date2num_list[ticker][k] - (volume_width/2), self.data_df.data['Volume'].values[k], width=volume_width, color=c, alpha=0.5)
            k += 1

        ax1v.plot(self.date2num_list[ticker][self.win_start:self.center_idx+1], self.ma_v[ticker][self.win_start:self.center_idx+1], 'y', linewidth=1.5)

        #ax1v.bar(date_num_list[-SP:], stock_disp['Volume'].values, color='#00ffe8')
        #ax1v.bar(date_num_list[-SP:], stock_disp['Volume'].values, colors)
        ax1v.axes.yaxis.set_ticklabels([])
        ax1v.grid(False)
        ax1v.set_ylim(0, 3 * self.data_df.data['Volume'].values[self.win_start:self.win_end + 1].max())
        ax1v.spines['bottom'].set_color("#5998ff")
        ax1v.spines['top'].set_color("#5998ff")
        ax1v.spines['left'].set_color("#5998ff")
        ax1v.spines['right'].set_color("#5998ff")
        ax1v.tick_params(axis='x', colors='w')
        ax1v.tick_params(axis='y', colors='w')


        '''
        #############################################################################################
        #try strategy
        middle, m_slope, std_min, std_min_n = qute_algo.adapt_lr(self.stock, self.center_idx-1)
        mid_a = middle[0]
        mid_b = middle[1]
        max_high = self.stock.data['High'].iloc[self.win_start:self.win_end+1].values.max()
        min_low = self.stock.data['Low'].iloc[self.win_start:self.win_end+1].values.min()

        mid_x_arr = []
        mid_y_arr = []
        mid_x = self.win_start
        while mid_x <= self.win_end:
            mid_y = mid_a * mid_x + mid_b
            if mid_y < max_high and mid_y > min_low:
               mid_x_arr.append(mid_x)
               mid_y_arr.append(mid_y)
            mid_x+=1

        #self.ax1.plot(mid_x_arr, mid_y_arr, 'y')
        if mid_y_arr:
            self.ax1.plot(self.date2num_list[ticker][mid_x_arr[0]:mid_x_arr[-1]+1], mid_y_arr, 'y--')
        #print('std_min = %f, std_min_n = %d, slope = %f, deviation = %f' % (std_min, std_min_n, m_slope, self.predict_values[ticker][self.center_idx]))
        #print('Adapt_lr slope: %f\n' % m_slope*100)

        # linear regression of volume
        v_period = 20
        v_x_arr = []
        v_y_arr = []
        v_idx = max(self.center_idx-v_period, self.win_start)
        v_x_max = min(self.center_idx+v_period, self.win_end)
        left = max(self.center_idx-v_period, 0)
        v_max_high = self.stock.data['Volume'].iloc[left:self.center_idx+v_period].values.max()
        v_min_low = self.stock.data['Volume'].iloc[left:self.center_idx+v_period].values.min()

        alr_par_v, alr_slope_v, alr_std_v, alr_period_v = qute_algo.adapt_lr1(self.stock.data['Volume'].values, self.center_idx, alr_start=20, max_alr_period=240)

        while v_idx < v_x_max:
            v_y = alr_par_v[0] * v_idx + alr_par_v[1]
            if v_y > v_min_low and v_y < v_max_high:
                v_x_arr.append(v_idx)
                v_y_arr.append(v_y)
            v_idx += 1

        ax1v.plot(self.date2num_list[ticker][v_x_arr[0]:v_x_arr[-1]+1], v_y_arr, 'y--', linewidth=1)

        '''


        '''
        #try lowess
        lowess, lowss_slope = qute_algo.cal_lowess(self.stock, self.center_idx, self.center_idx+1-self.win_start)
        #self.ax1.plot(self.date2num_list[ticker][int(lowess[:,0][0]):int(lowess[:,0][-1]+1)], lowess[:,1], 'w--')
        #print len(self.date2num_list[ticker][self.win_start+1:self.center_idx+1]), len(lowess[:,1][-(self.center_idx+1-self.win_start):-1])
        self.ax1.plot(self.date2num_list[ticker][self.win_start:self.center_idx+1], lowess[:,1][-(self.center_idx+1-self.win_start):], 'w--')
        lowess_start = -6
        lowess_end = -1
        lowess_slope = (lowess[:,1][lowess_end] - lowess[:,1][lowess_start]) / (lowess_end-lowess_start)
        lowess_slope1 = (lowess[:,1][-1] - lowess[:,1][-2])
        diff = self.sma5[ticker][self.center_idx] - lowess[:,1][-1]

        #print self.center_idx, self.stock.data['Date'].iloc[self.center_idx], lowess_slope1*240, diff, diff/self.sma5[ticker][self.center_idx]

        #try peaks_lr
        peaks_max, peaks_min, peaks, data = self.plr.find_peaks(self.stock, self.center_idx-1, before=40)
        cur_peak = self.plr.get_cur_peak(peaks_max, peaks_min)

        if peaks_max != []:
            for p in peaks_max:
                if p < self.win_start:
                    peaks_max.remove(p)
            if peaks_max != []:
                for p in peaks_max:
                    self.ax1.plot(self.date2num_list[ticker][p], self.stock.data['Close'].iloc[p], 'ro')

        if peaks_min != []:
            for p in peaks_min:
                if p < self.win_start:
                    peaks_min.remove(p)
            if peaks_min != []:
                for p in peaks_min:
                    self.ax1.plot(self.date2num_list[ticker][p], self.stock.data['Close'].iloc[p], 'go')

        if cur_peak != None:
            self.ax1.plot(self.date2num_list[ticker][cur_peak], self.stock.data['Close'].iloc[cur_peak], 'yo')



        max_high = self.stock.data['High'].iloc[self.win_start:self.win_end+1].values.max()
        min_low = self.stock.data['Low'].iloc[self.win_start:self.win_end+1].values.min()
        max_lr_par, max_slope, max_predict_err, max_period, max_std, min_lr_par, min_slope, min_predict_err, min_period, min_std, alr_par, alr_slope, alr_predict_err, alr_period, alr_std, alr_par_v, alr_slope_v, alr_std_v, alr_period_v= self.plr.run(self.stock, self.center_idx)

        max_x_arr = []
        max_y_arr = []
        max_x = self.win_start
        while max_x <= self.win_end:
            max_y = max_lr_par[0] * max_x + max_lr_par[1]
            if max_y > min_low and max_y < max_high:
               max_x_arr.append(max_x)
               max_y_arr.append(max_y)
            max_x+=1

        if max_x_arr != [] and max_y_arr != []:
            self.ax1.plot(self.date2num_list[ticker][max_x_arr[0]:max_x_arr[-1]+1], max_y_arr, 'g--')

        min_x_arr = []
        min_y_arr = []
        min_x = self.win_start
        while min_x <= self.win_end:
            min_y = min_lr_par[0] * min_x + min_lr_par[1]
            if min_y > min_low and min_y < max_high:
               min_x_arr.append(min_x)
               min_y_arr.append(min_y)
            min_x+=1

        if max_x_arr != [] and min_y_arr != []:
            self.ax1.plot(self.date2num_list[ticker][min_x_arr[0]:min_x_arr[-1]+1], min_y_arr, 'm--')

        '''



        ################################################################################################################
        ####KDJ
        self.ax0 = plt.subplot2grid((6,4), (0,0), sharex=self.ax1, rowspan=1, colspan=4, axisbg='#07000d')
        self.ax0.grid(True, color='w')
        if not self.hide_center_day:
            self.ax0.axvline(self.date2num_list[ticker][self.center_idx], ymax=(self.ax0.axis())[-1], color = center_idx_color, alpha=0.5)


        #self.ax0.plot(self.date2num_list[ticker][self.win_start:self.center_idx+1], self.efficiency[ticker][self.win_start:self.center_idx+1], 'b', linewidth=1.5)
        #self.ax0.plot(self.date2num_list[ticker][self.win_start:self.center_idx+1], self.alr_slope_v[ticker][self.win_start:self.center_idx+1], 'g', linewidth=1.5)
        #self.ax0.plot(self.date2num_list[ticker][self.win_start:self.center_idx+1], self.alr_slope_p[ticker][self.win_start:self.center_idx+1], 'y', linewidth=1.5)

        '''
        self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.slowk[ticker][self.win_start:self.win_end+1], 'b', linewidth=1.5)
        self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.slowd[ticker][self.win_start:self.win_end+1], 'g', linewidth=1.5)
        self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.slowj[ticker][self.win_start:self.win_end+1], 'r', linewidth=1.5)
        '''

        '''
        #self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], slope_list, 'r', linewidth=1.5)
        #self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], lowess_slope_list, 'g', linewidth=1.5)
        self.ax0.plot(self.date2num_list[ticker][self.win_start:self.center_idx], self.predict_values[ticker][self.win_start:self.center_idx], 'r', linewidth=1.5)
        self.ax0.axhline(0, color='b')
        self.ax0v = self.ax0.twinx()
        self.ax0v.plot(self.date2num_list[ticker][self.win_start:self.center_idx], (self.slope_list[ticker][self.win_start:self.center_idx])*1, 'g', linewidth=1.5)

        self.ax0.plot(self.date2num_list[ticker][self.win_start:self.center_idx], self.pr_sl[ticker][self.win_start:self.center_idx], 'y', linewidth=1.5)
        #print(self.predict_values[ticker][self.center_idx], self.slope_list[ticker][self.center_idx])
        '''

        #self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.obv[ticker][self.win_start:self.win_end+1], color='r')
        self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.ma_obv1[ticker][self.win_start:self.win_end+1], color='r')
        self.ax0.axhline(y=0, color='r')
        #self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.volume_glue_idx[ticker][self.win_start:self.win_end+1], color='c')

        self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.ma_obv[ticker][self.win_start:self.win_end+1], color='b')

        #self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.obv_glue[ticker][self.win_start:self.win_end+1], color='b')


        self.ax0.yaxis.label.set_color("w")
        #self.ax0.set_yticks([20,80])
        #self.ax0.axhline(80, color='r')
        #self.ax0.axhline(20, color='g')
        self.ax0.spines['bottom'].set_color("#5998ff")
        self.ax0.spines['top'].set_color("#5998ff")
        self.ax0.spines['left'].set_color("#5998ff")
        self.ax0.spines['right'].set_color("#5998ff")
        self.ax0.tick_params(axis='y', colors='w')
        self.ax0.tick_params(axis='x', colors='w')
        plt.ylabel('EXPERIMENTAL')


        self.ax2 = plt.subplot2grid((6,4), (5,0), sharex=self.ax1, rowspan=1, colspan=4, axisbg='#07000d')
        self.ax2.grid(True, color='w')
        if not self.hide_center_day:
            self.ax2.axvline(self.date2num_list[ticker][self.center_idx], ymax=(self.ax2.axis())[-1], color = center_idx_color, alpha=0.5)

        # MACD #
        '''
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.dif[ticker][self.win_start:self.win_end+1], color='red')
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.dem[ticker][self.win_start:self.win_end+1], color='green')
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.macdhist[ticker][self.win_start:self.win_end+1], color='yellow')
        #self.ax2.fill_between(self.date2num_list[ticker][self.win_start:self.win_end+1], dif[ticker][self.win_start:self.win_end+1], dem[ticker][self.win_start:self.win_end+1], where=(dif[ticker][self.win_start:self.win_end+1]<dem[ticker][self.win_start:self.win_end+1]), facecolor='g', edgecolor='g', alpha=0.5)
        self.ax2.fill_between(self.date2num_list[ticker][self.win_start:self.win_end+1], 0, self.macdhist[ticker][self.win_start:self.win_end+1], where=(self.dif[ticker][self.win_start:self.win_end+1] < self.dem[ticker][self.win_start:self.win_end+1]), facecolor='g', edgecolor='g', alpha=0.5)
        self.ax2.fill_between(self.date2num_list[ticker][self.win_start:self.win_end+1], 0, self.macdhist[ticker][self.win_start:self.win_end+1], where=(self.dif[ticker][self.win_start:self.win_end+1] >= self.dem[ticker][self.win_start:self.win_end+1]), facecolor='r', edgecolor='r', alpha=0.5)
        '''

        #glue index
        #self.ax2.set_ylim(-2, 2)
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.av_glue_idx[ticker][self.win_start:self.win_end+1], color='green')
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], (self.av_glue_idx[ticker][self.win_start:self.win_end+1] > 0)*0.5, color='green')
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], (self.av_glue_idx[ticker][self.win_start:self.win_end+1] < 0)*-0.5, color='green')
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.av_glue_idx1[ticker][self.win_start:self.win_end+1], color='c')

        #Raw Stochastic Value
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.rsv[ticker][self.win_start:self.win_end+1], color='y')

        #normalized OBV
        self.ax2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.nobv[ticker][self.win_start:self.win_end+1], color='m')
        #self.ax0.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.nobv[ticker][self.win_start:self.win_end+1]*0.0, color='m')
        self.ax2.axhline(y=0, color='m')



        '''
        self.ax2_2 = self.ax2.twinx()
        self.ax2_2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.ma_obv[ticker][self.win_start:self.win_end+1], color='b')
        self.ax2_2.plot(self.date2num_list[ticker][self.win_start:self.win_end+1], self.ma_obv[ticker][self.win_start:self.win_end+1]*0.0, color='b')
        '''


        max_x = np.argmax(self.av_glue_idx[ticker][self.win_start:self.win_end+1]) + self.win_start
        max_y = np.max(self.av_glue_idx[ticker][self.win_start:self.win_end+1])
        min_x = np.argmin(self.av_glue_idx[ticker][self.win_start:self.win_end+1]) + self.win_start
        min_y = np.min(self.av_glue_idx[ticker][self.win_start:self.win_end+1])
        if max_y == np.inf or max_y == -np.inf:
            max_y = 0
        if min_y == np.inf or min_y == -np.inf:
            min_y = 0
        #print max_x-self.win_start, max_y, min_x-self.win_start, min_y, self.win_start, self.date2num_list[ticker][self.win_start]
        self.ax2.plot(self.date2num_list[ticker][max_x], max_y, 'y+', self.date2num_list[ticker][min_x], min_y, 'yo')
        #self.ax2.plot(self.date2num_list[ticker][max_x], max_y, 'g+')

        '''
        slope_period = 20
        start = self.center_idx-slope_period
        end = self.center_idx
        self.price_slope = misc.cal_slope_angle(np.array(range(start,end)), self.stock_pool[ticker].data['Close'].iloc[start:end])
        #print self.stock_pool[ticker].data['Close'].iloc[start:end]
        #self.volume_slope = misc.cal_slope_angle(np.array(range(start,end)), self.ma_obv1[ticker][start:end]/1000000.0)
        '''



        #Display current index of prices and volumes
        #self.ax1.text(self.key_footnote, 0.98,  '%s\n%s\n%s\n%s\n%s\n%sM\n%s\n%sM\n%2.0f\n%2.0f' %(self.stock.data['Date'].values[self.center_idx], str(self.stock.data['Open'].values[self.center_idx])[:5], str(self.stock.data['High'].values[self.center_idx])[:5], str(self.stock.data['Low'].values[self.center_idx])[:5], str(self.stock.data['Close'].values[self.center_idx])[:5], str(round(self.stock.data['Volume'].values[self.center_idx]/1000000.0)), str(self.win_end-self.win_start+1), str(round(self.ma_obv1[ticker][self.center_idx]/1000000.0)), self.price_slope, self.volume_slope),
        self.ax1.text(self.key_footnote, 0.98,  '%s\n%s\n%s\n%s\n%s\n%sM\n%s\n%sM\n' % (self.data_df.data['Date'].values[self.center_idx], str(self.data_df.data['Open'].values[self.center_idx])[:5], str(self.data_df.data['High'].values[self.center_idx])[:5], str(self.data_df.data['Low'].values[self.center_idx])[:5], str(self.data_df.data['Close'].values[self.center_idx])[:5], str(round(self.data_df.data['Volume'].values[self.center_idx] / 1000000.0)), str(self.win_end - self.win_start + 1), str(round(self.ma_obv1[ticker][self.center_idx] / 1000000.0))),
                      fontsize=10, alpha=1.0,
                      horizontalalignment='left',
                      verticalalignment='top',
                      transform=self.ax1.transAxes, color=center_idx_color)

        #end indicator

        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
        self.ax2.spines['bottom'].set_color("#5998ff")
        self.ax2.spines['top'].set_color("#5998ff")
        self.ax2.spines['left'].set_color("#5998ff")
        self.ax2.spines['right'].set_color("#5998ff")
        self.ax2.tick_params(axis='x', colors='w')
        self.ax2.tick_params(axis='y', colors='w')
        self.ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
        #plt.ylabel('MACD')
        #self.ax2.set_ylabel('MACD')

        for label in self.ax2.xaxis.get_ticklabels():
            label.set_rotation(45)

        plt.suptitle(self.symbol.upper(), color='w')

        plt.setp(self.ax0.get_xticklabels(), visible=False)
        plt.setp(self.ax1.get_xticklabels(), visible=False)

        if self.hide_time:
            plt.setp(self.ax2.get_xticklabels(), visible=False)

        plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)


    def jump2next(self):

        #Define display window
        self.run_idx += 1
        if self.run_idx >= len(self.tagged_points_list):
            print('To the last!')
            self.run_idx = len(self.tagged_points_list) - 1

        self.symbol = self.tagged_points_list[self.run_idx][0]
        self.data_df = self.data_pool[self.symbol]
        self.action = int(self.tagged_points_list[self.run_idx][2])
        if self.tagged_points_list[self.run_idx][1] == '':
            self.center_idx = 0
        else:
            self.center_idx = self.data_df.time_str2idx(self.tagged_points_list[self.run_idx][1], type=2)

        if self.center_idx < 0:
            self.center_idx = 0

        #if self.candle_stick_data.has_key(self.ticker) == False:
        #    self.cal_all_data(self.ticker)

        self.cal_all_data(self.symbol)

        if self.center_idx < 0:
            print("Center day = %d." % self.center_idx)
            return

        self.__cal_win__()

    def jump2last(self):
        '''
        self.center_dates_idx -= 1
        if self.center_dates_idx < 0:
            print('To the first!')
            self.center_dates_idx = 0

        self.center_idx = self.stock.time2idx(self.center_dates[self.center_dates_idx], type=2)
        if self.center_idx < 0:
            print("Center day out of the range.")
            return

        self.__cal_win__()
        '''
        #Define display window
        self.run_idx -= 1
        if self.run_idx < 0:
            print('To the first!')
            self.run_idx = 0

        self.symbol = self.tagged_points_list[self.run_idx][0]
        self.data_df = self.data_pool[self.symbol]
        self.action = int(self.tagged_points_list[self.run_idx][2])
        self.center_idx = self.data_df.time_str2idx(self.tagged_points_list[self.run_idx][1], type=2)

        if self.center_idx < 0:
            self.center_idx = 0

        #if self.candle_stick_data.has_key(self.ticker) == False:
        #    self.cal_all_data(self.ticker)
        self.cal_all_data(self.symbol)

        if self.center_idx < 0:
            print("Center day = %d." % self.center_idx)
            return

        self.__cal_win__()


    def save2files(self):
        time_str = now2str()
        if self.trade_record:
            file_name = 'tr_' +  time_str + '.log'
            file = open(self.mydir.trade_record + file_name, 'w')
            file.writelines(self.trade_record)
            file.close()

        k = 0
        while k < 10:
            if self.mark_list[k]:
                file_name = 'm_' + str(k) + '_' +  time_str + '.dat'
                file = open(self.mydir.pos + file_name, 'w')
                file.writelines(self.mark_list[k])
                file.close()
            k+=1

        print((self.htf_list))
        if self.htf_list:
            save2pjson(self.htf_list, self.mydir.training_original + 'tmp.htf')
            print('Save to tmp.htf')


    def run(self, info, symbols):
        self.tagged_points_list = []
        if type(info) == tuple:
            self.tagged_points_list.append(info)
        elif type(info) == list:
            self.tagged_points_list = info
        else:
            print('Wrong input ticker_time in Visual.run.')
            exit()

        self.symbols = symbols
        self.symbol = self.symbols[0]
        self.data_df = self.data_pool[self.symbol]
        self.action = int(self.tagged_points_list[self.run_idx][2])

        #Generate tagged points
        self.tagged_points = {}
        for e in self.tagged_points_list:
            if (e[0] in self.tagged_points) == False:
                self.tagged_points[e[0]] = {}
                self.tagged_points[e[0]]['time'] = []
                self.tagged_points[e[0]]['action'] = []
            #date_num = mdates.date2num(dt.datetime.strptime(e[1], '%Y-%m-%d'))
            self.tagged_points[e[0]]['time'].append(e[1])
            self.tagged_points[e[0]]['action'].append(e[2])

        #Init figure
        #self.fig = plt.figure()
        #plt.switch_backend('TkAgg')
        plt.switch_backend('QT5Agg')
        self.fig = plt.figure(facecolor='#07000d')
        mng = plt.get_current_fig_manager()
        #mng.window.state('zoomed')
        #mng.full_screen_toggle()
        mng.window.showMaximized()  #Maxmize the figure window

        #connect win
        #self.fig.canvas.mpl_connect('pick_event', self.on_press)
        #cid = self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        #self.fig.canvas.setFocusPolicy( Qt.ClickFocus )
        #self.fig.canvas.setFocus()

        on_move_id = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        cid = self.fig.canvas.mpl_connect('button_release_event', self.on_click)

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        #self.fig.canvas.mpl_connect('key_press_event', self.on_pick)

        '''
        self.center_dates = []
        if type(time) == str:
            self.center_dates.append(time)
        elif type(time) == list:
            self.center_dates = time
        self.center_dates_idx = 0

        if self.center_dates == None:
            print('Invalid date.')
            return
        '''
        #self.tagged_set = set(self.ticker)
        self.cal_all_data(self.symbol)

        self.run_idx = -1
        self.jump2next()
        self.draw(self.symbol, self.action)

        if self.plt_show == 0:
            plt.show()
            self.plt_show = 1

        #save to files
        self.save2files()
        print ('Game over!')

    def key_process(self, key):
        #show/hide the vertical line
        if key == 'h':
            self.hide_center_day = 1 - self.hide_center_day
            self.__step_center__(0)

        if key == 'y':
            self.hide_circle = 1 - self.hide_circle
            self.__step_center__(0)

        if key == 't':
            self.hide_time = 1 - self.hide_time
            self.__step_center__(0)

        # Moving
        elif key == 'left' or key == 'a':
            step = -1
            self.__step_center__(step)
        elif key == 'right' or key == 'd':
            step = 1
            self.__step_center__(step)
        elif key == 'pageup':
            step = self.win_start - self.center_idx
            self.__step_center__(step)
        elif key == 'pagedown':
            step = self.win_end - self.center_idx
            self.__step_center__(step)

        #Moving the window
        elif key == 'ctrl+left':
            step = -1
            self.__step_win__(step)
        elif key == 'ctrl+right':
            step = 1
            self.__step_win__(step)

        #Fast moving
        elif key == 'alt+left':
            step = int((self.win_start - self.win_end)/3)
            self.__step_win__(step)
            print('Left: ', self.left_time)

        elif key == 'alt+right':
            step = int((self.win_end - self.win_start)/3)
            self.__step_win__(step)
            print('Left: ', self.left_time)

        elif key == 'c':
            self.key_footnote, self.mouse_footnote = self.mouse_footnote, self.key_footnote
            self.__step_win__(0)

        #browse the markers
        elif key == 'n':
            self.jump2next()
        elif key == 'N':
            self.jump2last()

        #Zooming
        elif key == 'up':
            self.before = int(self.before/self.zoom_ratio)
            self.after = int(self.after/self.zoom_ratio)
            self.__cal_win__()
        elif key == 'down':
            self.before = int(self.before*self.zoom_ratio)
            self.after = int(self.after*self.zoom_ratio)
            self.__cal_win__()

        #buy and sell
        elif key == 'i' or key == 'e':
            self.cur_time = self.data_df.data['Date'].iloc[self.center_idx]
            print('Buy %s on %s' % (self.symbol, self.cur_time))
            tmp_str = self.symbol + ',' + self.cur_time + ',1\n'
            print(tmp_str)
            self.trade_record.append(tmp_str)
            self.undo_keys.append('i')

        elif key == 'o':
            self.cur_time = self.data_df.data['Date'].iloc[self.center_idx]
            print('Sell %s on %s' % (self.symbol, self.cur_time))
            tmp_str = self.symbol + ',' + self.cur_time + ',-1\n'
            print(tmp_str)
            self.trade_record.append(tmp_str)
            self.undo_keys.append('o')

        elif key == u'u':
            undo = self.undo_keys.pop()
            print('Undo %s' % undo)
            if undo == 'i' or undo == 'o':
                self.trade_record.pop()
            elif undo >= '0' and undo <= '9':
                self.mark_list[int(undo)].pop()

        #bottom
        elif key == '1':
            row = OrderedDict()
            row['ticker'] = self.symbol
            row['start'] = self.left_time
            row['end'] = self.right_time
            row['tag'] = 1
            self.htf_list.append(row)
            print('ticker: %s, start: %s, end: %s, tag: %d\n' % (row['ticker'], row['start'], row['end'], row['tag']))

        #top
        elif key == '6':
            row = OrderedDict()
            row['ticker'] = self.symbol
            row['start'] = self.left_time
            row['end'] = self.right_time
            row['tag'] = -1
            self.htf_list.append(row)
            print('ticker: %s, start: %s, end: %s, tag: %d\n' % (row['ticker'], row['start'], row['end'], row['tag']))

        #up
        elif key == '3':
            row = OrderedDict()
            row['ticker'] = self.symbol
            row['start'] = self.left_time
            row['end'] = self.right_time
            row['tag'] = 2
            self.htf_list.append(row)
            print('ticker: %s, start: %s, end: %s, tag: %d\n' % (row['ticker'], row['start'], row['end'], row['tag']))

        #down
        elif key == '8':
            row = OrderedDict()
            row['ticker'] = self.symbol
            row['start'] = self.left_time
            row['end'] = self.right_time
            row['tag'] = -2
            self.htf_list.append(row)
            print('ticker: %s, start: %s, end: %s, tag: %d\n' % (row['ticker'], row['start'], row['end'], row['tag']))

        #unknow trend, no trend
        elif key == '0':
            row = OrderedDict()
            row['ticker'] = self.symbol
            row['start'] = self.left_time
            row['end'] = self.right_time
            row['tag'] = 0
            self.htf_list.append(row)
            print('ticker: %s, start: %s, end: %s, tag: %d\n' % (row['ticker'], row['start'], row['end'], row['tag']))

        elif key == 'R':
            if self.htf_list:
                row = self.htf_list.pop()
                print('Cancel: ticker: %s, start: %s, end: %s, tag: %d\n' % (row['ticker'], row['start'], row['end'], row['tag']))
            else:
                print('htf_list is empty!')



            #print(self.left_time, self.right_time)
        '''
        #mark data
        elif key >= '0' and key <= '9':
            self.cur_time = self.stock.data['Date'].iloc[self.center_idx]
            tmp_str = self.ticker + ',' + self.cur_time + '\n'
            print('In %s mark %s' % (key, tmp_str))
            self.mark_list[int(key)].append(tmp_str)
            self.undo_keys.append(key)
        '''


        #positive and negative training set

    #mouse click event
    def on_click(self, event):
        print('on click')
        print(event.xdata, event.ydata, event.x, event.y)

        if (event.xdata):
            time_str = dt2str(mdates.num2date(event.xdata))
            self.left_time = time_str
            time_num = self.data_df.time_str2idx(time_str, type=1)
            if time_num < 0:
                print('No time index here.')
                return

            self.center_idx = time_num
            self.draw(self.symbol, self.action)
            plt.draw()

    def on_move(self, event):
        #print('on move')
        #print(event.xdata, event.ydata, event.x, event.y)
        if (event.xdata):
            self.draw(self.symbol, self.action)
            time_str = dt2str(mdates.num2date(event.xdata))
            self.right_time = time_str
            time_num = self.data_df.time_str2idx(time_str, type=1)

            # Text: price, volume...
            if time_num > 0:
                self.ax1.text(self.mouse_footnote, 0.98,  '%s\n%s\n%s\n%s\n%s\n%sM\n%s' % (self.data_df.data['Date'].values[time_num], str(self.data_df.data['Open'].values[time_num])[:5], str(self.data_df.data['High'].values[time_num])[:5], str(self.data_df.data['Low'].values[time_num])[:5], str(self.data_df.data['Close'].values[time_num])[:5], str(round(self.data_df.data['Volume'].values[time_num] / 1000000.0)), str(self.win_end - self.win_start + 1)),
                              fontsize=10, alpha=1.0,
                              horizontalalignment='left',
                              verticalalignment='top',
                              transform=self.ax1.transAxes, color='y')

            #draw circle
            #radius, circle_color = self.draw_circle(self.ticker, time_num)
            #self.ax1.plot(self.date2num_list[self.ticker][time_num], self.candle_stick_data[self.ticker][time_num][2], '.', color=circle_color, markersize=radius, alpha=0.4)

            #draw vertical line
            self.ax1.axvline(event.xdata, ymax=(self.ax1.axis())[-1], color = 'y', alpha=0.5)
            self.ax0.axvline(event.xdata, ymax=(self.ax0.axis())[-1], color = 'y', alpha=0.5)
            self.ax2.axvline(event.xdata, ymax=(self.ax2.axis())[-1], color = 'y', alpha=0.5)

            plt.draw()
            #f = plt.figure()
            #f.canvas.draw()

    #Check keyboard event
    def on_press(self, event):
        print('press', event.key)
        sys.stdout.flush()
        plt.clf()

        self.key_process(event.key)
        self.draw(self.symbol, self.action)
        plt.draw()

    #Check keyboard event
    def on_press1(self, event):
        print('press', event.key)
        sys.stdout.flush()
        plt.clf()

        self.key_process(event.key)
        self.draw(self.symbol, self.action)

        self.fig.canvas.blit()
        curpic = self.fig.canvas.copy_from_bbox()

        plt.draw()

def run(symbol):
    start = '2014-01-01'
    data_dict = load_data([symbol], start=start, verbose=True)

    visual = Visual(data_dict, './', before=240)
    visual.run((symbol, start, 1), [symbol])


if __name__ == '__main__':
    args = sys.argv[1:]
    symbol = args[0]
    import cProfile
    #cProfile.run('run()', 'profile.log')
    run(symbol)

