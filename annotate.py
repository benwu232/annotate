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
    global win_start, win_end, symbol
    tag2color(tag_line, tag_colors, win_start, win_end)
    ax1.clear()
    ax2.clear()
    fb = ax2.fill_between(x_line[win_start:win_end+1], volume_line[win_start:win_end+1], color='gray', linewidth=1)
    ax2.set_ylim(0, 4 * volume_line[win_start:win_end+1].max())

    #ax1.plot(x_line[win_start:win_end+1], price_line[win_start:win_end+1], 'gray', label='Price', linewidth=1)
    line = ax1.plot(x_line[win_start:win_end+1], price_line[win_start:win_end+1], 'gray', linewidth=1)
    #ax1.plot(x_line[win_start:win_end+1], price_line[win_start:win_end+1], color='gray', marker='o', linewidth=1)
    scatter = ax1.scatter(x_line[win_start:win_end+1], price_line[win_start:win_end+1], c=tag_colors[win_start:win_end+1], s=50)
    plt.title(symbol, color='w')
    ax2.grid(True, color='w')
    #ax1.tick_params(axis='x', colors='w')
    ax1.tick_params(axis='y', colors='w')
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')

    ax1.set_xlim(win_start, win_end)
    ax2.set_xlim(win_start, win_end)

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

def on_press(event):
    global symbol, win_start, win_end, is_multi_tag, multi_tag_start
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
        save_tag(tag_dir, symbol)

    elif event.key == 'h':
        print('0-unknown, 1-buy, 2-sell, 3-short, 4-cover, 5-up, 6-down')
        #print('0-gray, 1-red, 2-green, 3-m, 4-cover, 5-up, 6-down')

    elif event.key == 'left':
        step_draw(-1)

    elif event.key == 'right':
        step_draw(1)

    elif event.key == 'ctrl+left':
        step = (win_end - win_start) // 4
        step_draw(-step)

    elif event.key == 'ctrl+right':
        step = (win_end - win_start) // 4
        step_draw(step)


def on_click(event):
    global win_start, win_end, seq_len
    if not event.inaxes:
        return

    print(ax1.get_xlim())
    print('on click')
    print(event.xdata, event.ydata, event.x, event.y)

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

    return


######################################################################
args = sys.argv[1:]
symbol = args[0]
data_dict = load_data([symbol], start='2014-01-01', verbose=True)
data_dict = datapro(data_dict)
data_df = data_dict[symbol]

time_line = pd.to_datetime(data_df.index//2, unit='s').values
x_line = list(range(len(time_line)))
price_line = data_df['LastPrice'].values
#price_line = np.nan_to_num(price_line)
volume_line = data_df['Volume'].values
#volume_line = np.nan_to_num(volume_line)
seq_len = len(price_line)
tag_file = os.path.join(tag_dir, symbol) + '.tag'
if os.path.isfile(tag_file):
    with open(tag_file) as fp:
        tag_line = np.asarray(json.load(fp), dtype=np.int8)
else:
    tag_line = np.zeros_like(price_line, dtype=np.int8)
tag_colors = ['' for _ in range(seq_len)]
win_start = 0
win_end = 50
is_multi_tag = False
multi_tag_start = 0


fig, ax2 = plt.subplots(facecolor='#07000d')
ax1 = ax2.twinx()
ax2.set_facecolor('#07000d')

fig.canvas.mpl_connect('key_press_event', on_press)
cid = fig.canvas.mpl_connect('button_release_event', on_click)
#cursor = Cursor(ax1, useblit=True, color='#9ffeb0', linewidth=1)
cursor = Cursor(ax1, useblit=True, color='cyan', linewidth=1)
#cursor = Cursor(ax1)
#plt.connect('motion_notify_event', cursor.mouse_move)

draw()
plt.show()
save_tag(tag_dir, symbol)

