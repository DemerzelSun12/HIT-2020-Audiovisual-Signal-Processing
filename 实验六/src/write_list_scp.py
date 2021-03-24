import os
import numpy as np


def write():
    file_name = '../list1.scp'
    with open(file_name, 'w+', encoding='utf8') as f:
        for i in range(1, 6):
            f.write(f'd:\data\wav\model\data{i}.wav d:\data\mfcc\model\data{i}.mfc\n')
            for j in range(1, 11):
                f.write(f'd:\data\wav\\test\data{i}_{j}.wav d:\data\mfcc\\test\data{i}_{j}.mfc\n')


if __name__ == '__main__':
    write()
