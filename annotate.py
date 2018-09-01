
import argparse
from lib.core import *


if __name__ == '__main__':
    futures_dir = '/home/wb/work/qute/data/futures_symbol/'

    parser = argparse.ArgumentParser(description='Annotation and visualization.')
    parser.add_argument('symbol', type=str, help='Symbol name')
    parser.add_argument('-d', dest='data_dir', default=futures_dir, help='Futures data directory')
    parser.add_argument('-w', type=int, dest='win_size', default=800, help='Window size to show')
    #parser.add_argument('-s', action='store', dest='simple_value', help='Store a simple value')
    args = parser.parse_args()

    #key_in = ['cu1502']
    #key_in = ['cu1502', '-s', 'fiel', '-d', '1234']
    #args = parser.parse_args(key_in)
    #print(args)
    #exit()

    futures_dir = args.data_dir
    symbol = args.symbol
    win_size = args.win_size

    #with open(futures_data_file, 'rb') as f:
    #    data_dict = pickle.load(f)

    anno = Annotation(symbol, futures_dir, win_size)
    anno.run()

# example: python annotate.py cu1409 -w 1000 -d /home/wb/work/qute/data/ini/futures_data.pkl


