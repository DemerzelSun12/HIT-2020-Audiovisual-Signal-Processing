from read_wav_from_file import *
from write_to_file import *
from functools import reduce
import math

zero_txt = '_zero.txt'
energy_txt = '_en.txt'


class EnergyZeroCrossingRateExtract(object):

    def __init__(self, wav_data_dic):
        self.wav_data_dic = wav_data_dic
        self.__frame_length = 256
        self.energy_dic = None
        self.zero_dic = None

    def calculate_energy(self):
        energy_dic = {}
        for key in self.wav_data_dic.keys():
            print(f'calculate energy of wav {key}.')
            sample_number = len(self.wav_data_dic[key][4])
            frame_number = math.ceil(sample_number / self.__frame_length)
            energy_of_key = []
            start = 0
            for i in range(frame_number):
                end = min(start + self.__frame_length, sample_number)
                sum_ = reduce(lambda x, y: x + y, (
                    map(lambda x: pow(x, 2) * pow(1.0 / self.__frame_length, 2), self.wav_data_dic[key][4][start:end])))
                # print(sum_)
                start += self.__frame_length
                energy_of_key.append(sum_)
            energy_dic[key] = energy_of_key
        return energy_dic

    def calculate_zero_crossing_rate(self):
        zero_rate = {}
        for key in self.wav_data_dic.keys():
            print(f'calculate zero crossing rate of wav {key}.')
            sample_number = len(self.wav_data_dic[key][4])
            frame_number = math.ceil(sample_number / self.__frame_length)
            start = 0
            zero_rate_of_key = []
            for i in range(frame_number):
                sum_ = 0
                end = min(start + self.__frame_length, sample_number)
                for j in range(start, end):
                    sum_ += int(self.wav_data_dic[key][4][j]) * int(self.wav_data_dic[key][4][j - 1]) < 0
                start += self.__frame_length
                zero_rate_of_key.append(sum_ / (self.__frame_length - 1))
            zero_rate[key] = zero_rate_of_key
        return zero_rate

    def energy_main(self):
        energy_dic = self.calculate_energy()
        self.energy_dic = energy_dic
        return energy_dic

    def zero_main(self):
        zero_dic = self.calculate_zero_crossing_rate()
        self.zero_dic = zero_dic
        return zero_dic


if __name__ == '__main__':
    dic = read_wav_from_file()
    energy = EnergyZeroCrossingRateExtract(dic)
    energy_dic_result = energy.energy_main()
    write_txt_to_file(energy_dic_result, energy_txt)
    zero_dic_result = energy.zero_main()
    write_txt_to_file(zero_dic_result, zero_txt)
