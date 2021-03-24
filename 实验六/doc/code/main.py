from read_mfcc_from_file import *
from dynamic_tme_warping import *


def main():
    model_dic = read_mfcc_from_file('model')
    test_dic = read_mfcc_from_file('test')
    dtw = DTW(model_dic, test_dic)
    dtw.dtw_main()


if __name__ == '__main__':
    main()
