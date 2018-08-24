"""
Show how to connect to keypress events
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from lib.common import *

tag_dir = 'tag'

def save2pjson(src, out_file):
    src_list = []
    for n in src:
        src_list.append(n.item())

    with open(out_file, 'w') as f:
          json.dump(src_list, f, ensure_ascii=False, indent='')

def save_tag(tag_dir, symbol):
    tag_file = '{}/{}.tag'.format(tag_dir, symbol)
    print('Save to {}'.format(tag_file))
    save2pjson(list(tag_line), tag_file)

def draw():
    global win_start, win_end, symbol_match, minute_span
    tag2color(tag_line, tag_colors, win_start, win_end)
    ax1.clear()
    ax2.clear()
    ax2.fill_between(x_line[win_start:win_end+1], volume_line[win_start:win_end+1], color='gray', linewidth=1, alpha=0.5)
    ax2.set_ylim(0, 4 * volume_line[win_start:win_end+1].max())

    #ax1.plot(x_line[win_start:win_end+1], price_line[win_start:win_end+1], 'gray', label='Price', linewidth=1)
    ax1.plot(x_line[win_start:win_end+1], price_line[win_start:win_end+1], 'gray', linewidth=1)
    #ax1.plot(x_line[win_start:win_end+1], price_line[win_start:win_end+1], color='gray', marker='o', linewidth=1)
    ax1.scatter(x_line[win_start:win_end+1], price_line[win_start:win_end+1], c=tag_colors[win_start:win_end+1], s=50)
    ax1.axvline(minute_span, ymax=(ax1.axis())[-1], color='green', alpha=0.5)
    #scatter.set_edgecolors(tag_colors[win_start:win_end+1])
    #scatter.set_offsets(np.c_[x_line[win_start:win_end+1], price_line[win_start:win_end+1]])
    #scatter.set_sizes(np.ones_like(price_line) * 50)

    plt.title(title_str)
    ax2.grid(True)

    #ax1.tick_params(axis='y')
    #ax2.tick_params(axis='x')
    #ax2.tick_params(axis='y')

    ax1.set_xlim(win_start, win_end)
    ax2.set_xlim(win_start, win_end)
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

def step_win(step):
    global win_start, win_end

    win_end += step
    win_start += step

    if win_start < 0:
        win_start = 0
    elif win_start >= seq_len:
        win_start = seq_len - 1

    if win_end < 0:
        win_end = 0
    elif win_end >= seq_len:
        win_end = seq_len - 1

def step_draw(step):
    step_win(step)
    draw()

def set_tag(x, y):
    global tag_line
    print('Set ({}, {})'.format(x, y))
    tag_line[x] = y

def set_tags(x_start, x_end, y):
    global tag_line, is_multi_tag, multi_tag_start
    if is_multi_tag:
        print('Multi tag end at {}'.format(x_end))
        print('Set x[{}:{}] to {}'.format(x_start, x_end, y))
        tag_line[x_start:x_end+1] = y

def get_ratio():
    global win_start, win_end, price_line
    ratio = price_line[win_start:win_end+1].max() / price_line[win_start:win_end+1].min()
    ratio_str = 'max / min = {}'.format(ratio)
    return ratio_str

def gen_formated_time_str(time_str):
    formated_str = '{} {}'.format(time_str[:10], time_str[11:])
    return formated_str

def get_info(event):
    x_int = int(round(event.xdata))
    time_str = (time_str_line[x_int])
    print('Current: {},    {} {},     {}'.format(x_int, time_str[:10], time_str[11:], price_line[x_int]))
    print('Time range {} - {}'.format(gen_formated_time_str(time_str_line[win_start]), gen_formated_time_str(time_str_line[win_end])))
    print('Price range {} - {}, {}'.format(price_line[win_start:win_end+1].min(), price_line[win_start:win_end+1].max(), get_ratio()))

def on_press(event):
    global symbol_match, win_start, win_end, is_multi_tag, multi_tag_start
    print('\npress', event.key)
    sys.stdout.flush()

    if event.key in ['0', '1', '2', '3', '4', '5', '6']:
        if is_multi_tag:
            set_tags(x_start=multi_tag_start, x_end=int(round(event.xdata)), y=int(event.key))
            is_multi_tag = False
        else:
            set_tag(int(round(event.xdata)), int(event.key))
        draw()

    elif event.key == '[':
        multi_tag_start = int(round(event.xdata))
        set_tag(multi_tag_start, 7)
        print('Multi tag start at {}'.format(multi_tag_start))
        is_multi_tag = True
        draw()

    elif event.key == 'c':
        save_tag(tag_dir, symbol_match)

    elif event.key == 't':
        x_int = int(round(event.xdata))
        time_str = (time_str_line[x_int])
        print('{} ===>> {} {}'.format(x_int, time_str[:10], time_str[11:]))

    elif event.key == 'i':
        get_info(event)

    elif event.key == 'h':
        print('0-unknown, 1-buy, 2-sell, 3-short, 4-cover, 5-up, 6-down')
        print('c-save, i-information, [-multi-tag start')
        print('left-window left 1 step, right-window right 1 step')
        print('ctrl+left-window left 1/4, ctrl+right-window right 1/4')
        print('scroll up - zoom out, scroll down - zoom in')

    elif event.key == 'left':
        step_draw(-1)
        get_ratio()

    elif event.key == 'right':
        step_draw(1)
        get_ratio()

    elif event.key == 'ctrl+left':
        step = (win_end - win_start) // 4
        step_draw(-step)
        get_ratio()

    elif event.key == 'ctrl+right':
        step = (win_end - win_start) // 4
        step_draw(step)
        get_ratio()


def on_click(event):
    global win_start, win_end, seq_len, title_str
    if not event.inaxes:
        return

    #print(ax1.get_xlim())
    print('\non click')
    #print(event.xdata, event.ydata, event.x, event.y)
    x_int = int(round(event.xdata))
    time_str = (time_str_line[x_int])
    title_str = '{}    {}'.format(symbol_match, gen_formated_time_str(time_str))

    win_left, win_right = ax1.get_xlim()
    win_start = int(win_left + 1)
    win_end = int(win_right)

    if win_start < 0:
        win_start = 0
    elif win_start >= seq_len:
        win_start = seq_len - 1

    if win_end < 0:
        win_end = 0
    elif win_end >= seq_len:
        win_end = seq_len - 1

    #margin = int((win_end - win_start) * 0.05)
    #win_start += margin
    #win_end -= margin

    draw()
    get_info(event)

def on_scroll(event):
    global win_start, win_end, seq_len
    zoom_rate = 1.1
    if not event.inaxes:
        return

    #print(ax1.get_xlim())
    print('\non scroll')
    #print(event.xdata, event.ydata, event.x, event.y)

    win_left, win_right = ax1.get_xlim()
    cursor_x = int(round(event.xdata))
    dif_left = cursor_x - int(round(win_left))
    dif_right = int(round(win_right)) - cursor_x

    if event.button == 'up':
        dif_left = int(dif_left // zoom_rate)
        dif_right = int(dif_right // zoom_rate)
        win_start = cursor_x - dif_left
        win_end = cursor_x + dif_right

    elif event.button == 'down':
        dif_left = int(dif_left * zoom_rate)
        dif_right = int(dif_right * zoom_rate)
        win_start = cursor_x - dif_left
        win_end = cursor_x + dif_right

    if win_start < 0:
        win_start = 0
    elif win_start >= seq_len:
        win_start = seq_len - 1

    if win_end < 0:
        win_end = 0
    elif win_end >= seq_len:
        win_end = seq_len - 1

    #margin = int((win_end - win_start) * 0.05)
    #win_start += margin
    #win_end -= margin

    draw()
    get_info(event)



######################################################################
args = sys.argv[1:]
symbol_match = args[0]
pre_offset = 4096
avg_period = 1
minute_span = pre_offset // avg_period
info_file = '/home/wb/work/qute/data/ini/futures_dict.pkl'
data_dir = '/home/wb/work/data/futures_data/'
with open(info_file, 'rb') as f:
    futures_info = pickle.load(f)
futures_dict = sel_active_futures(futures_info, active_distance_rate_min=2.0, active_variation_min=4000, active_span_size=10, day_pre_offset=15)
data_dict = load_futures_data(data_dir, futures_dict, symbol_match, start='2014-01-01', end='2015-06-01', pre_offset=pre_offset, verbose=True)
#data_dict = load_futures_data([symbol], start='2014-01-01', pre_tick_offset=pre_tick_offset, avg_period=avg_period, verbose=True)
data_dict = datapro(data_dict)
data_df = data_dict[symbol_match]

#time_line = pd.to_datetime(data_df.index//2+8*3600, unit='s').values
time_line = pd.to_datetime(data_df.index+8*3600, unit='s').values
time_str_line = [str(item)[:16] for item in time_line]
x_line = list(range(len(time_line)))
price_line = data_df['LastPrice'].values
#price_line = np.nan_to_num(price_line)
volume_line = data_df['Volume'].values
#volume_line = np.nan_to_num(volume_line)
seq_len = len(price_line)
tag_file = os.path.join(tag_dir, symbol_match) + '.tag'
if os.path.isfile(tag_file):
    with open(tag_file) as fp:
        tag_line = np.asarray(json.load(fp), dtype=np.int8)
else:
    tag_line = np.zeros_like(price_line, dtype=np.int8)
tag_colors = ['' for _ in range(seq_len)]
win_start = minute_span - 30
win_end = minute_span + 30
title_str = symbol_match

plt.style.use('dark_background')
fig, ax2 = plt.subplots(facecolor='#07000d')
ax1 = ax2.twinx()
ax2.set_facecolor('#07000d')
plt.title(symbol_match)
#scatter = ax1.scatter(x_line[win_start:win_end+1], price_line[win_start:win_end+1], s=50)
ax1.spines['bottom'].set_color("#5998ff")
ax1.spines['top'].set_color("#5998ff")
ax1.spines['left'].set_color("#5998ff")
ax1.spines['right'].set_color("#5998ff")

fig.canvas.mpl_connect('key_press_event', on_press)
cid = fig.canvas.mpl_connect('button_release_event', on_click)
fig.canvas.mpl_connect('scroll_event', on_scroll)
#cursor = Cursor(ax1, useblit=True, color='#9ffeb0', linewidth=1)
cursor = Cursor(ax1, useblit=True, color='cyan', linewidth=1)
#cursor = Cursor(ax1)
#plt.connect('motion_notify_event', cursor.mouse_move)

draw()
plt.show()
save_tag(tag_dir, symbol_match)

