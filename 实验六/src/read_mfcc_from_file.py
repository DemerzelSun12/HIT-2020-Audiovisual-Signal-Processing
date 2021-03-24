import os
import struct
import re


def read_mfcc_from_file(wav_type):
    file_dir_path = f'../data/mfcc/{wav_type}/'
    file_list = os.listdir(file_dir_path)
    wav_data_dic = {}
    for file in file_list:
        file_path = os.path.join(file_dir_path, file)
        # print("open mfcc file " + file_path)
        file_name = str(re.split("[dat.]", file)[4])
        # wav_type = file_name.split('_')[0] # 类别，在对比时使用
        # print(file_name)
        with open(file_path, "rb") as f:
            n_frames = struct.unpack('>i', f.read(4))[0]
            frame_rate = struct.unpack('>i', f.read(4))[0]
            feature_byte = struct.unpack('>h', f.read(2))[0]
            feature_kind = struct.unpack('>h', f.read(2))[0]
            # print(feature_byte)
            data_of_file = []
            for i in range(n_frames):
                data_feature = []
                for j in range(int(feature_byte / 4)):
                    data_feature.append(struct.unpack('>f', f.read(4))[0])
                data_of_file.append(data_feature)
            wav_data_dic[file_name] = data_of_file
    # print(wav_data_dic.keys())
    return wav_data_dic


def main():
    read_mfcc_from_file("model")
    read_mfcc_from_file("test")


if __name__ == '__main__':
    main()
