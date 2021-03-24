from scipy.io import wavfile
import wave
import numpy as np
import struct


class Decode:
    data = []  # 读入的采样点
    encode_eight = []  # 8-bit编码时解码的数据
    eight_decode = []  # 8-bit记录一边解码的数据
    decode_eight = []  # 8-bit解码的数据
    encode_four = []  # 4-bit编码的数据
    four_decode = []  # 4-bit一边编码一边解码的数据
    decode_four = []  # 4-bit解码的数据

    def __init__(self):
        fs, self.data = wavfile.read('../data/1.wav')  # 读入数据

    # 4-bits 编码
    def encode_four_bits(self, a):
        num = 0
        n = len(self.data)
        # 先编码第一个数
        # 小于0
        if self.data[0] < 0:
            if - self.data[0] < a:  # 如果小于量化因子，直接编码为-1
                num = 15
            else:
                num = 15 - (np.uint8(int(0.5 + abs(np.log(- self.data[0] / a))) << 4) >> 4)  # 当数据为负时，符号位设置为1
            if num < 8:  # 下溢，编码为8
                self.encode_four.append(8)
            else:  # 正常
                self.encode_four.append(num)
            self.four_decode.append(-a * np.exp((15 - self.encode_four[0])))  # 负数,乘以量化因子a
        # 大于等于0
        else:
            if self.data[0] < a:  # 小于量化因子，直接编码为1
                num = 0
            else:
                num = np.unit8(int(0.5 + abs(np.log(self.data[0] / a))) << 4) >> 4
            if num > 7:  # 上溢，编码为7
                self.encode_four.append(7)
            else:  # 正常
                self.encode_four.append(num)
            self.four_decode.append(a * np.exp(self.encode_eight[0]))  # 正数解码，乘以量化因子a

        # 编码剩余的点
        for i in range(1, n):
            num = 0
            # 与解码后的数据相减的差
            d = self.data[i] - self.four_decode[i - 1]
            if d < 0:  # 差为负
                if -d < a:  # 小于量化因子
                    num = 15
                else:
                    num = 15 - (np.uint8(int(0.5 + abs(np.log(-d / a))) << 4) >> 4)  # 当数据为负时，符号位设置为1
                if num < 8:  # 下溢
                    self.encode_four.append(8)
                else:
                    self.encode_four.append(num)
                self.four_decode.append(self.four_decode[i - 1] - a * np.exp((15 - self.encode_four[i])))  # 负数,乘以量化因子a
            else:  # 差非负
                if d < a:  # 小于量化因子
                    num = 0
                else:
                    num = np.uint8(int(0.5 + abs(np.log(d / a + 1e-5))) << 4) >> 4
                if num >= 8:  # 上溢
                    self.encode_four.append(7)
                else:
                    self.encode_four.append(num)
                self.four_decode.append(self.four_decode[i - 1] + a * np.exp(self.encode_four[i]))
        # print(self.encode_four)

        f = open('1_4bit.dpc', 'wb')  # 打开写入文件
        # 防止奇数个时溢出，故减一
        length = len(self.encode_four)
        for i in range(0, length - 1, 2):
            # 两个一起写
            x = (self.encode_four[i] << 4) + self.encode_four[i + 1]
            # print(np.uint8(x))
            f.write(np.uint8(x))
        if length % 2 == 1:
            x = self.encode_four[length - 1] << 4
            f.write(np.uint8(x))
        f.close()

    # 4-bits解码
    def decode_four_bits(self, a):
        f = open('1_4bit.dpc', 'rb')
        cnt = -1
        while True:
            ff = f.read(1)
            if not ff:
                break
            ff = struct.unpack('B', ff)
            ff = ff[0]
            # print(ff)
            # 第一个数的解析
            if len(self.decode_four) == 0:
                r = 0
                x1 = ff >> 4  # 前4-bit
                x2 = ff & 15  # 后4-bit
                # print(x1, x2)
                if x1 < 8:  # 符号位为0，则为正数
                    r = a * np.exp(x1)
                else:
                    r = -a * np.exp(15 - int(x1))
                self.decode_four.append(r)

                if x2 < 8:  # 符号位为0，则为正数
                    r = self.decode_four[0] + a * np.exp(x2)
                else:
                    r = self.decode_four[0] - a * np.exp(15 - int(x2))
                self.decode_four.append(r)
                cnt += 2
                # print(self.decode_four[0])

            else:
                x1 = ff >> 4  # 前4-bit
                x2 = ff & 15  # 后4-bit
                # print(x1,x2)
                # print(self.decode_four[cnt])
                if x1 < 8:  # 符号位为0，则为正数
                    r = self.decode_four[cnt] + a * np.exp(x1)
                else:
                    r = self.decode_four[cnt] - a * np.exp(15 - int(x1))
                self.decode_four.append(r)
                # print(r)
                cnt += 1
                # print(self.decode_four[cnt])
                if x2 < 8:  # 符号位为0，则为正数
                    r = self.decode_four[cnt] + a * np.exp(x2)
                else:
                    r = self.decode_four[cnt] - a * np.exp(15 - int(x2))
                self.decode_four.append(r)
                # print(r)
                cnt += 1

        f.close()
        # 计算信噪比
        print(cal_snr(self.data, self.decode_four))
        # print(self.decode_four)
        # 写入文件
        f = wave.open('1_4bit.pcm', 'wb')
        f.setnchannels(1)  # 配置声道数
        f.setsampwidth(2)  # 配置量化位数
        f.setframerate(16000)  # 配置取样频率
        f.writeframes(np.array(self.decode_four).astype(np.short).tostring())  # 转换为二进制数据写入文件
        f.close()

    # 8-bits 编码
    def encode_eight_bits(self):
        num = 0
        n = len(self.data)
        f = open('1_8bit.dpc', 'wb')  # 打开写入文件
        if self.data[0] < 0:
            num = 255 - np.uint8(0.5 + abs(np.log(-self.data[0])))  # 当数据为负时，符号位设置为1
        if num == 0:
            self.encode_eight.append(np.uint8(0.5 + abs(np.log(self.data[0]))))
            self.eight_decode.append(np.exp(self.encode_eight[0]))  # 正数
        else:
            if num < 128:
                self.encode_eight.append(np.uint8(128))
            else:
                self.encode_eight.append(np.uint8(num))
            self.eight_decode.append(-np.exp(255 - int(self.encode_eight[0])))  # 负数
        # print(self.encode_eight[0], self.eight_decode[0])

        f.write(self.encode_eight[0])  # 写入文件
        for i in range(1, n):
            num = 0
            # 与解码后的数据相减
            if self.data[i] - self.eight_decode[i - 1] < 0:
                num = 255 - np.uint8(0.5 + abs(np.log(-self.data[i] + self.eight_decode[i - 1])))
                if num < 128:
                    self.encode_eight.append(np.uint8(128))
                else:
                    self.encode_eight.append(np.uint8(num))
                self.eight_decode.append(self.eight_decode[i - 1] - np.exp(255 * 1.0 - self.encode_eight[i]))

            elif self.data[i] - self.encode_eight[i - 1] == 0:
                self.encode_eight.append(np.uint8(num))
                self.eight_decode.append(self.eight_decode[i - 1])

            else:
                num += np.uint8(0.5 + abs(np.log(1e-5 + self.data[i] - 1.0 * self.eight_decode[i - 1])))
                if num >= 128:
                    self.encode_eight.append(np.uint8(127))
                else:
                    self.encode_eight.append(np.uint8(num))
                self.eight_decode.append(self.eight_decode[i - 1] + np.exp(1.0 * self.encode_eight[i]))

            # 将第i次编码写入文件
            f.write(self.encode_eight[i])
        f.close()
        # print(self.encode_eight)

    # 8-bit解码,存入数组中,并写入文件
    def decode_eight_bits(self):
        f = open("1_8bit.dpc", "rb")
        cnt = -1
        while True:
            ff = f.read(1)
            if not ff:
                break
            ff = struct.unpack('B', ff)
            ff = ff[0]
            print(ff)
            # 第一个数的解析
            if len(self.decode_eight) == 0:
                r = 0
                if ff < 128:  # 符号位为0，则为正数
                    r = np.exp(ff)
                else:
                    r = -np.exp(255 - int(ff))
                self.decode_eight.append(r)
                print(r)
                # print('re_num:'+str(re_num))
            else:
                re_num = int(ff) & int(np.uint8(128))  # 如果为0，说明为正数或0；不为0则为负数
                # print('re_num:'+str(re_num))
                r = 0
                if re_num == 0:
                    r = np.exp(ff)
                else:
                    r = -np.exp(255 - int(ff))
                r += self.decode_eight[cnt]
                print(r)
                self.decode_eight.append(r)
            cnt += 1

        f.close()
        # 计算信噪比
        print(cal_snr(self.data, self.decode_eight))
        # 写入文件
        f = wave.open('1_8bit.pcm', 'wb')
        f.setnchannels(1)  # 配置声道数
        f.setsampwidth(2)  # 配置量化位数
        f.setframerate(16000)  # 配置取样频率
        f.writeframes(np.array(self.decode_eight).astype(np.short).tostring())  # 转换为二进制数据写入文件
        f.close()


# 计算信噪比
def cal_snr(before, after):
    x = np.longlong(0)
    y = np.longlong(0)
    length = len(before)
    for i in range(length):
        # print('before:'+str(before[i]))
        # print('after:'+str(after[i]))
        x += before[i] ** 2 / length  # 防止溢出
        y += (before[i] - after[i]) ** 2 / length  # 防止溢出
        # print('x:'+str(x))
        # print('y:'+str(y))
    # print(x)
    # print(y)
    return 10 * np.log10(x / y)


test = Decode()
test.encode_eight_bits()  # 8-bit
test.decode_eight_bits()
a = 25
# test.encode_four_bits(a)  # 4-bit
# test.decode_four_bits(a)
