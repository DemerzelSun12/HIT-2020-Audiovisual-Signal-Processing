import os

file_dir_path = '../result/'


def write_end_point_file(wav_data_dic, end_point_dic, pcm):
    for key in end_point_dic.keys():
        # print(end_point_dic[key])
        print(f'write end point detection of wav {key}.')
        file_name = f'result_{key}' + pcm
        with open(file_dir_path + file_name, 'wb') as f:
            i = 0
            while i < len(end_point_dic[key]):
                for location in wav_data_dic[key][4][end_point_dic[key][i] * 256:end_point_dic[key][i + 1] * 256]:
                    # print(f'print {location}')
                    f.write(location)
                i += 2


def write_txt_to_file(wav_energy_dic, txt):
    for key in wav_energy_dic.keys():
        file_name = f'{key}' + txt
        print(f'write {txt.split("_")[1]} of wav {key}.')
        with open(file_dir_path + file_name, 'w', encoding='utf8') as f:
            for i in range(len(wav_energy_dic[key])):
                f.write(f'{wav_energy_dic[key][i]}\n')
