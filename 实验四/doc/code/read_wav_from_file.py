import os
import wave
import numpy as np

zero_txt = '_zero.txt'
energy_txt = '_en.txt'


def read_wav_from_file():
    file_dir_path = '../data/'
    file_list = os.listdir(file_dir_path)
    wav_data_dic = {}
    for file in file_list:
        file_path = os.path.join(file_dir_path, file)
        print("open wav file " + file_path)
        file_name = int(file.split('.', 1)[0])
        # print(file_name)
        with wave.open(file_path, "rb") as f:
            params = f.getparams()
            n_channels, samp_width, frame_rate, n_frames = params[:4]
            wav_data = f.readframes(n_frames)
            wave_data = np.frombuffer(wav_data, dtype=np.short)
            wav_data_dic[file_name] = [n_channels, samp_width, frame_rate, n_frames, wave_data]
    # print(wav_data_dic.keys())
    return wav_data_dic


def main():
    read_wav_from_file()


if __name__ == '__main__':
    main()
