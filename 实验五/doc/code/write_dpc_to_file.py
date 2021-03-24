import numpy as np

file_dir_path = '../result/dpc/'
suffix = '.dpc'


def write_dpc_to_file(code_dic):
    for key in code_dic.keys():
        print(f'write dpc code with 4 bits.')
        file_name = '1_4bit' + suffix
        with open(file_dir_path + file_name, 'wb') as f:
            length = len(code_dic[key])
            for i in range(0, length - 1, 2):
                data = (code_dic[key][i] << 4) + code_dic[key][i + 1]
                # print(np.uint8(data))
                f.write(np.uint8(data))
            if length % 2 == 1:
                data = code_dic[key][length - 1] << 4
                f.write(np.uint8(data))


def write_8_dpc_to_file(code_dic):

    for key in code_dic.keys():
        print(f'write dpc code with 8 bits.')
        file_name = '1_8bit' + suffix
        with open(file_dir_path + file_name, 'wb') as f:
            length = len(code_dic[key])
            for i in range(length):
                f.write(code_dic[key][i])
