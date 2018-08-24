
import argparse
from lib.core import *


if __name__ == '__main__':
    futures_data_file = '/home/wb/work/qute/data/runtime/futures_data.pkl'

    parser = argparse.ArgumentParser(description='Annotation and visualization.')
    parser.add_argument('symbol', type=str, help='Symbol name')
    parser.add_argument('-d', dest='data_file', default=futures_data_file, help='Futures data dict file')
    #parser.add_argument('-s', action='store', dest='simple_value', help='Store a simple value')
    args = parser.parse_args()

    #key_in = ['cu1502']
    #key_in = ['cu1502', '-s', 'fiel', '-d', '1234']
    #args = parser.parse_args(key_in)
    #print(args)
    #exit()

    futures_data_file = args.data_file
    symbol = args.symbol

    with open(futures_data_file, 'rb') as f:
        data_dict = pickle.load(f)

    anno = Annotation(data_dict[symbol])
    anno.run()



