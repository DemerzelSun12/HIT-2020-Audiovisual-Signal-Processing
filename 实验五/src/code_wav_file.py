from src.write_dpc_to_file import *
from src.read_wav_from_file import *
from src.calculate_snr import *
import numpy as np
import struct


class Codewavfile(object):
    def __init__(self, wav_dic):
        self.wav_dic = wav_dic
        self.__quantization_factor = 23
        self.code_dic_4 = {}
        self.code_dic_8 = {}

    def difference_pulse_coding_modulation_4(self):
        print('DPCM 4bits')
        for key in self.wav_dic.keys():
            code_data = np.zeros_like(self.wav_dic[key][4])
            decode_data = np.zeros_like(self.wav_dic[key][4])
            length = len(self.wav_dic[key][4])
            code_data[0], decode_data[0] = self.coding_first_4(self.wav_dic[key][4][0])
            for i in range(1, length):
                err = self.wav_dic[key][4][i] - decode_data[i - 1]
                code_data[i] = self.dpcm_4(err)
                decode_data[i] = decode_data[i - 1] - self.__quantization_factor * np.exp(
                    15 - code_data[i]) if err < 0 else decode_data[i - 1] + self.__quantization_factor * np.exp(
                    code_data[i])
            self.code_dic_4[key] = code_data
            # print(code_data)

    def coding_first_4(self, data):
        """
        对第一位进行4 bits编码
        :param data: 第一位数据
        :return: 第一位数据编码结果与解码结果
        """
        if data < 0:
            code_data = 15 if data > -self.__quantization_factor else 15 - (
                    np.uint8(int(abs(np.log(-data / self.__quantization_factor)) + 0.5) << 4) >> 4)
            code_data = max(8, code_data)
            return code_data, np.exp(15 - code_data) * -self.__quantization_factor
        else:
            code_data = 0 if data < self.__quantization_factor else np.uint8(
                int(abs(np.log(data / self.__quantization_factor)) + 0.5) << 4) >> 4
            code_data = min(7, code_data)
            return code_data, np.exp(code_data) * self.__quantization_factor

    def dpcm_4(self, data):
        if data < 0:
            code_data = 15 if data > -self.__quantization_factor else 15 - (
                    np.uint8(int(abs(np.log(-data / self.__quantization_factor)) + 0.5) << 4) >> 4)
            code_data = max(8, code_data)
            # print(code_data)
            return code_data
        else:
            code_data = 0 if data < self.__quantization_factor else np.uint8(
                int(abs(np.log(data / self.__quantization_factor)) + 0.5) << 4) >> 4
            code_data = min(7, code_data)
            # print(code_data)
            return code_data

    def difference_pulse_coding_modulation_8(self):
        print('DPCM 8bits')
        for key in self.wav_dic.keys():
            code_data = []
            decode_data = []
            length = len(self.wav_dic[key][4])
            a, b = self.coding_first_8(self.wav_dic[key][4][0])
            code_data.append(a)
            decode_data.append(b)
            # print(code_data[0], decode_data[0])
            for i in range(1, length):
                err = self.wav_dic[key][4][i] - decode_data[i - 1]
                err2 = self.wav_dic[key][4][i] - code_data[i - 1]
                if err < 0:
                    code_i = 255 - np.uint8(abs(np.log(-err)) + 0.5)
                    code_i = np.uint8(128) if code_i < 128 else np.uint8(code_i)
                    code_data.append(code_i)
                    decode_data.append(decode_data[i - 1] - np.exp(255 * 1.0 - code_data[i]))
                elif err2 == 0:
                    code_data.append(np.uint8(0))
                    decode_data.append(decode_data[i - 1])
                else:
                    code_i = np.uint8(0.5 + abs(np.log(1e-5 + self.wav_dic[key][4][i] - 1.0 * decode_data[i - 1])))
                    code_i = np.uint8(127) if code_i >= 128 else np.uint8(code_i)
                    code_data.append(code_i)
                    decode_data.append(decode_data[i - 1] + np.exp(1.0 * code_data[i]))
            self.code_dic_8[key] = code_data
            # for i in range(len(code_data)): print(code_data[i])

    def coding_first_8(self, data):
        code_data = 255 - np.uint8(abs(np.log(-data / self.__quantization_factor)) + 0.5) if data < 0 else np.uint8(
            abs(np.log(data / self.__quantization_factor)) + 0.5)
        code_data = np.uint8(128) if code_data < 128 else np.uint8(code_data)
        if data < 0:
            # print(code_data, -np.exp(255 - int(code_data)) * -self.__quantization_factor)
            return code_data, -np.exp(255 - int(code_data)) * -self.__quantization_factor
        else:
            # print(f'2code data {code_data}, decode {np.exp(code_data) * self.__quantization_factor}')
            return code_data, np.exp(code_data) * self.__quantization_factor

    def dpcm_8(self, data):
        if data < 0:
            code_data = 255 - np.uint8(abs(np.log(-data)) + 0.5)
            code_data = np.uint8(128) if code_data < 128 else np.uint8(code_data)
            return code_data
        elif data == 0:
            code_data = np.uint8(data)
            return code_data

    def decode_4_bits(self):
        for key in self.code_dic_4.keys():
            file_name = f'../result/dpc/{key}_4bit.dpc'
            decode_four = []
            with open(file_name, 'rb') as f:
                num = -1
                while True:
                    data = f.read(1)
                    if not data: break
                    data = struct.unpack('B', data)
                    data = data[0]
                    # print(data)
                    if len(decode_four) == 0:
                        high_4 = data >> 4
                        low_4 = data & 15
                        # print(high_4, low_4)
                        read = self.__quantization_factor * np.exp(
                            high_4) if high_4 < 8 else -self.__quantization_factor * np.exp(15 - int(high_4))
                        decode_four.append(read)
                        read = decode_four[0] + self.__quantization_factor * np.exp(low_4) if low_4 < 8 else \
                            decode_four[0] - self.__quantization_factor * np.exp(15 - int(low_4))
                        decode_four.append(read)
                        num += 2
                        # print(decode_four[0])
                    else:
                        high_4 = data >> 4
                        low_4 = data & 15
                        # print(high_4, low_4)
                        # print(decode_four[num])
                        read = decode_four[num] + self.__quantization_factor * np.exp(high_4) if high_4 < 8 else \
                            decode_four[num] - self.__quantization_factor * np.exp(15 - int(high_4))
                        decode_four.append(read)
                        # print(read)
                        num += 1
                        # print(decode_four[num])
                        read = decode_four[num] + self.__quantization_factor * np.exp(low_4) if low_4 < 8 else \
                            decode_four[num] - self.__quantization_factor * np.exp(15 - int(low_4))
                        decode_four.append(read)
                        # print(read)
                        num += 1
            calculate_snr(self.wav_dic[key][4], decode_four)
            # print(decode_four)
            file_write_name = f'../result/pcm/{key}_4bit.pcm'
            with wave.open(file_write_name, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(np.array(decode_four).astype(np.short).tostring())

    def decode_8_bits(self):
        for key in self.code_dic_8.keys():
            file_name = f'../result/dpc/{key}_8bit.dpc'
            decode_eight = []
            with open(file_name, 'rb') as f:
                num = -1
                while True:
                    data = f.read(1)
                    if not data: break
                    data = struct.unpack("B", data)
                    data = data[0]
                    # print(data)
                    if len(decode_eight) == 0:
                        read = np.exp(data) if data < 128 else -np.exp(255 - int(data))
                        # print(read)
                        decode_eight.append(read)
                    else:
                        data_code = int(data) & int(np.uint8(128))
                        read = np.exp(data) if data_code == 0 else -np.exp(255 - int(data))
                        read += decode_eight[num]
                        # print(read)
                        decode_eight.append(read)
                    num += 1
            calculate_snr(self.wav_dic[key][4], decode_eight)
            file_write_name = f'../result/pcm/{key}_8bit.pcm'
            with wave.open(file_write_name, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(np.array(decode_eight).astype(np.short).tostring())


def main():
    wav_dic = read_wav_from_file()
    code_wav_file = Codewavfile(wav_dic)
    code_wav_file.difference_pulse_coding_modulation_4()
    code_wav_file.difference_pulse_coding_modulation_8()
    code_4_dic = code_wav_file.code_dic_4
    code_8_dic = code_wav_file.code_dic_8
    write_dpc_to_file(code_4_dic)
    write_8_dpc_to_file(code_8_dic)
    code_wav_file.decode_4_bits()
    code_wav_file.decode_8_bits()


if __name__ == '__main__':
    main()
