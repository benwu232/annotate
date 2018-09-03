"""
Show how to connect to keypress events
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.widgets import Cursor

from lib.common import *

label_dir = 'label'



def save2pjson(src, out_file):
    src_list = []
    for n in src:
        src_list.append(n.item())

    with open(out_file, 'w') as f:
          json.dump(src_list, f, ensure_ascii=False, indent='')




class Annotation(object):
    def __init__(self, symbol, data_dir, win_size=800):
        self.data_dict_file = data_dir + symbol + '.pkl'
        with open(self.data_dict_file, 'rb') as f:
            self.data_dict = pickle.load(f)
        self.symbol = self.data_dict['name']
        self.df = self.data_dict['df']
        self.pre_offset = self.data_dict['pre_offset']

        self.win_start = self.pre_offset - win_size//2
        self.win_end = self.pre_offset + win_size//2
        self.price_line = self.df['LastPrice'].values
        self.volume_line = self.df['Volume'].values
        self.seq_len = len(self.price_line)
        self.x_line = list(range(self.seq_len))
        if 'label' in self.data_dict:
            self.label_line = self.data_dict['label']
        else:
            self.label_line = np.zeros(self.seq_len)
        self.label_colors = ['' for _ in range(self.seq_len)]

        self.time_line = pd.to_datetime(self.df.index+8*3600, unit='s').values
        self.time_str_line = [str(item)[:16] for item in self.time_line]
        self.title_str = self.symbol
        self.is_multi_label = False

        self.last_x = self.cur_x = self.pre_offset
        self.last_y = self.cur_y = self.price_line[self.cur_x]
        self.restore_stack = []

    def run(self):
        plt.style.use('dark_background')
        self.fig, self.ax2 = plt.subplots(facecolor='#07000d')
        self.ax1 = self.ax2.twinx()
        self.ax2.set_facecolor('#07000d')
        plt.title(self.title_str)
        #scatter = ax1.scatter(x_line[win_start:win_end+1], price_line[win_start:win_end+1], s=50)
        self.ax1.spines['bottom'].set_color("#5998ff")
        self.ax1.spines['top'].set_color("#5998ff")
        self.ax1.spines['left'].set_color("#5998ff")
        self.ax1.spines['right'].set_color("#5998ff")

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        #dbclick = self.fig.canvas.mpl_connect('button_press_event', self.on_click2)
        click = self.fig.canvas.mpl_connect('button_release_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        #cursor = Cursor(ax1, useblit=True, color='#9ffeb0', linewidth=1)
        self.cursor = Cursor(self.ax1, useblit=True, color='cyan', linewidth=1)
        #cursor = Cursor(ax1)
        #plt.connect('motion_notify_event', cursor.mouse_move)

        self.draw()
        plt.show()
        #self.save_label(label_dir, self.symbol)
        save_dump(self.data_dict, self.data_dict_file)

    def draw(self):
        label2color(self.label_line, self.label_colors, self.win_start, self.win_end)
        self.ax1.clear()
        self.ax2.clear()
        self.ax2.fill_between(self.x_line[self.win_start:self.win_end+1], self.volume_line[self.win_start:self.win_end+1], color='gray', linewidth=1, alpha=0.5)
        self.ax2.set_ylim(0, 8 * np.mean(self.volume_line[self.win_start:self.win_end+1]))

        #ax1.plot(x_line[win_start:win_end+1], price_line[win_start:win_end+1], 'gray', label='Price', linewidth=1)
        self.ax1.plot(self.x_line[self.win_start:self.win_end+1], self.price_line[self.win_start:self.win_end+1], 'gray', linewidth=1)
        #ax1.plot(x_line[win_start:win_end+1], price_line[win_start:win_end+1], color='gray', marker='o', linewidth=1)
        self.ax1.scatter(self.x_line[self.win_start:self.win_end+1], self.price_line[self.win_start:self.win_end+1], c=self.label_colors[self.win_start:self.win_end+1], s=50)
        self.ax1.axvline(self.pre_offset, ymax=(self.ax1.axis())[-1], color='green', alpha=0.5)

        plt.title(self.title_str)
        self.ax2.grid(True)

        #ax1.tick_params(axis='y')
        #ax2.tick_params(axis='x')
        #ax2.tick_params(axis='y')

        self.ax1.set_xlim(self.win_start, self.win_end)
        self.ax2.set_xlim(self.win_start, self.win_end)
        #ax2.text(0.9, 0.98,  time_text,
        #                    fontsize=10, alpha=1.0,
        #                    horizontalalignment='left',
        #                    verticalalignment='top',
        #                    transform=ax2.transAxes, color='y')

        #ax1.yaxis.label.set_color("w")
        #ax1.xaxis.label.set_color("w")
        #ax2.yaxis.label.set_color("w")

        #fig.canvas.draw()
        plt.draw()

    def step_win(self, step):
        self.win_end += step
        self.win_start += step

        if self.win_start < 0:
            self.win_start = 0
        elif self.win_start >= self.seq_len:
            self.win_start = self.seq_len - 1

        if self.win_end < 0:
            self.win_end = 0
        elif self.win_end >= self.seq_len:
            self.win_end = self.seq_len - 1

    def step_draw(self, step):
        self.step_win(step)
        self.draw()

    def set_label(self, x, y):
        #print('Set ({}, {})'.format(x, y))
        self.label_line[x] = y

    def set_labels(self, x_start, x_end, y):
        #print('Set x[{}:{}] = {}'.format(x_start, x_end, y))
        self.label_line[x_start:x_end+1] = y

    def get_ratio(self):
        ratio = self.price_line[self.win_start:self.win_end+1].max() / self.price_line[self.win_start:self.win_end+1].min()
        ratio_str = 'max / min = {}'.format(ratio)
        return ratio_str

    def gen_formated_time_str(self, time_str):
        formated_str = '{} {}'.format(time_str[:10], time_str[11:])
        return formated_str

    def get_info(self, event):
        self.x_int = int(round(event.xdata))
        self.time_str = (self.time_str_line[self.x_int])
        print('Available length: {}, from {} to {}'.format(len(self.price_line)-self.pre_offset, self.gen_formated_time_str(self.time_str_line[self.pre_offset]), self.gen_formated_time_str(self.time_str_line[-1])))
        print('Current: {},    {} {},     {}'.format(self.x_int, self.time_str[:10], self.time_str[11:], self.price_line[self.x_int]))
        print('Time range {} - {}'.format(self.gen_formated_time_str(self.time_str_line[self.win_start]), self.gen_formated_time_str(self.time_str_line[self.win_end])))
        print('Price range {} - {}, {}'.format(self.price_line[self.win_start:self.win_end+1].min(), self.price_line[self.win_start:self.win_end+1].max(), self.get_ratio()))

    def get_click_info(self, event):
        print('x: {} -> {}, y: {} -> {}'.format(self.last_x, self.cur_x, self.last_y, self.cur_y))
        print('x_range: {}, y_range: {}, change_rate: {}'.format(self.cur_x-self.last_x, self.cur_y-self.last_y, self.cur_y/self.last_y-1))

    def save_label(self, label_dir, symbol):
        label_file = '{}/{}.label'.format(label_dir, symbol)
        print('Save to {}'.format(label_file))
        save2pjson(list(self.label_line), label_file)

    def on_press(self, event):
        print('\npress', event.key)
        sys.stdout.flush()

        if event.key in ['0', '1', '2', '3', '4', '5', '6']:
            if self.is_multi_label:
                x_end = int(round(event.xdata))
                self.restore_stack.append((self.multi_label_start, copy.copy(self.label_line[self.multi_label_start:x_end+1])))
                self.set_labels(x_start=self.multi_label_start, x_end=x_end, y=int(event.key))
                print('Multi label ] = {}, label = {}'.format(int(round(event.xdata)), event.key))
                self.is_multi_label = False
            else:
                x = int(round(event.xdata))
                self.restore_stack.append((x, copy.copy(self.label_line[x])))
                self.set_label(x, int(event.key))
                print('x = {}, label = {}'.format(int(round(event.xdata)), event.key))
            self.draw()

        elif event.key == 'u':
            if len(self.restore_stack) > 0:
                x_start, values = self.restore_stack.pop()
                if type(values) is not np.ndarray:
                    values = [values]
                for k, value in enumerate(values):
                    self.set_label(x_start+k, value)
            self.draw()

        #elif event.key == '[':
        #    self.multi_label_start = int(round(event.xdata))
        #    self.set_label(self.multi_label_start, 7)
        #    print('Multi label start at {}'.format(self.multi_label_start))
        #    self.is_multi_label = True
        #    self.draw()

        elif event.key == 'c':
            #self.save_label(label_dir, self.symbol)
            save_dump(self.data_dict, self.data_dict_file)

        #elif event.key == 't':
        #    x_int = int(round(event.xdata))
        #    time_str = (self.time_str_line[x_int])
        #    print('{} ===>> {} {}'.format(x_int, time_str[:10], time_str[11:]))

        elif event.key == 'i':
            self.get_info(event)

        elif event.key == 'h':
            print('0-unknown, 1-buy, 2-sell, 3-short, 4-cover, 5-up, 6-down')
            print('c-save, i-information, right_click-start multi-label, u-undo labeling')
            print('left-window left 1 step, right-window right 1 step')
            print('ctrl+left-window left 1/4, ctrl+right-window right 1/4')
            print('scroll up - zoom out, scroll down - zoom in')

        elif event.key == 'left':
            self.step_draw(-1)
            self.get_ratio()

        elif event.key == 'right':
            self.step_draw(1)
            self.get_ratio()

        elif event.key == 'ctrl+left':
            step = (self.win_end - self.win_start) // 4
            self.step_draw(-step)
            self.get_ratio()

        elif event.key == 'ctrl+right':
            step = (self.win_end - self.win_start) // 4
            self.step_draw(step)
            self.get_ratio()

    def on_click2(self, event):
        if not event.inaxes:
            return

        if not event.dblclick:
            return
        print('Double click')


    def on_click(self, event):
        if not event.inaxes:
            return

        #print(ax1.get_xlim())
        print('\non click')
        #print(event.xdata, event.ydata, event.x, event.y)
        #right button, start multi-label
        if event.button == 3:
            self.multi_label_start = int(round(event.xdata))
            #self.set_label(self.multi_label_start, 7)
            #print('Multi label start at {}'.format(self.multi_label_start))
            print('Multi label [ = {}'.format(self.multi_label_start))
            self.is_multi_label = True
            #self.draw()
        y_min, y_max = self.ax1.get_ylim()
        y_per = (y_max / y_min - 1)
        self.last_x = self.cur_x
        self.last_y = self.cur_y
        self.cur_x = int(round(event.xdata))
        self.cur_y = self.price_line[self.cur_x]
        time_str = (self.time_str_line[self.cur_x])
        self.title_str = '{}    {}    {:.4f}'.format(self.symbol, self.gen_formated_time_str(time_str), y_per)

        self.win_left, self.win_right = self.ax1.get_xlim()
        self.win_start = int(self.win_left + 1)
        self.win_end = int(self.win_right)

        if self.win_start < 0:
            self.win_start = 0
        elif self.win_start >= self.seq_len:
            self.win_start = self.seq_len - 1

        if self.win_end < 0:
            self.win_end = 0
        elif self.win_end >= self.seq_len:
            self.win_end = self.seq_len - 1

        #margin = int((win_end - win_start) * 0.05)
        #win_start += margin
        #win_end -= margin

        self.draw()
        self.get_click_info(event)

    def on_scroll(self, event):
        zoom_rate = 1.1
        if not event.inaxes:
            return

        #print(ax1.get_xlim())
        print('\non scroll')
        #print(event.xdata, event.ydata, event.x, event.y)

        self.win_left, self.win_right = self.ax1.get_xlim()
        cursor_x = int(round(event.xdata))
        dif_left = cursor_x - int(round(self.win_left))
        dif_right = int(round(self.win_right)) - cursor_x

        if event.button == 'up':
            dif_left = int(dif_left // zoom_rate)
            dif_right = int(dif_right // zoom_rate)
            self.win_start = cursor_x - dif_left
            self.win_end = cursor_x + dif_right

        elif event.button == 'down':
            dif_left = int(dif_left * zoom_rate)
            dif_right = int(dif_right * zoom_rate)
            self.win_start = cursor_x - dif_left
            self.win_end = cursor_x + dif_right

        if self.win_start < 0:
            self.win_start = 0
        elif self.win_start >= self.seq_len:
            self.win_start = self.seq_len - 1

        if self.win_end < 0:
            self.win_end = 0
        elif self.win_end >= self.seq_len:
            self.win_end = self.seq_len - 1

        #margin = int((win_end - win_start) * 0.05)
        #win_start += margin
        #win_end -= margin

        self.draw()
        self.get_info(event)




