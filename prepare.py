
from lib.common import *


if __name__ == '__main__':
    data_dict = load_data()
    symbol_list = list(data_dict.keys())
    symbol_list.sort()
    with open('symbol_list.json', 'w') as outfile:
        json.dump(symbol_list, outfile)

