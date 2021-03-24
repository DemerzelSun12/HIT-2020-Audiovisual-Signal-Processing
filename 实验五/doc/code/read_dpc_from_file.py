import os
import struct


def read_dpc_from_file():
    file_dir_path = '../result/dpc/'
    file_list = os.listdir(file_dir_path)
    wav_data_dic = {}
    for file in file_list:
        file_path = os.path.join(file_dir_path, file)
        print("open dpc file " + file_path)
        file_name = int(file.split('.', 1)[0])
        # print(file_name)
        dpc_data = []
        with open(file_path, "rb") as f:
            data = struct.unpack('h', f.read(2))
            dpc_data.append(data[0])
            while True:
                data_byte = f.read(1)
                if not data_byte:
                    break
                else:
                    data = struct.unpack('B', data_byte)
                    dpc_data.append(data[0])

    return wav_data_dic


def main():
    read_dpc_from_file()


if __name__ == '__main__':
    main()
