import energy_zero_crossing_rate_extract

pcm_file = '.pcm'


class EndPointDetect(object):

    def __init__(self, energy_dic, zero_dic):
        self.energy_dic = energy_dic
        self.zero_dic = zero_dic

    def end_point_detect(self):
        end_point_dic = {}
        for key in self.energy_dic.keys():
            print(f'end point detection of  wav {key}.')
            average_energy = sum(self.energy_dic[key]) / len(self.energy_dic[key])
            threshold_low = sum(self.energy_dic[key][:10]) / 10
            # 能量的高阈值
            threshold_high = average_energy / 4
            # 能量的低阈值
            threshold_low = (threshold_low + threshold_high) / 4
            # 过零率的阈值
            threshold_zero_cross = sum(self.zero_dic[key][:10]) / 10
            flag = False
            threshold_high_test = []
            threshold_low_test = []
            threshold_zero_cross_test = []
            for i in range(len(self.energy_dic[key])):
                if (len(threshold_high_test) == 0 and bool(1 - flag) and self.energy_dic[key][i] > threshold_high) or (
                        bool(1 - flag) and self.energy_dic[key][i] > threshold_high and i - 30 > threshold_high_test[
                    len(threshold_high_test) - 1]):
                    threshold_high_test.append(i)
                    flag = bool(1 - flag)
                elif bool(1 - flag) and self.energy_dic[key][i] > threshold_high and i - 30 <= threshold_high_test[
                    len(threshold_high_test) - 1]:
                    threshold_high_test = threshold_high_test[:len(threshold_high_test) - 1]
                    flag = True

                if flag and self.energy_dic[key][i] < threshold_high:
                    threshold_high_test.append(i)
                    flag = False
            for j in range(len(threshold_high_test)):
                test = threshold_high_test[j]
                if j % 2 == 1:
                    while test < len(self.energy_dic[key]) and self.energy_dic[key][test] > threshold_low: test += 1
                else:
                    while test > 0 and self.energy_dic[key][test] > threshold_low: test -= 1
                threshold_low_test.append(test)
            for j in range(len(threshold_low_test)):
                test = threshold_low_test[j]
                if j % 2 == 1:
                    while test < len(self.zero_dic[key]) and self.zero_dic[key][
                        test] >= 3 * threshold_zero_cross: test += 1
                else:
                    while test > 0 and self.zero_dic[key][test] >= 3 * threshold_zero_cross: test -= 1
                threshold_zero_cross_test.append(test)

            print(f'end point detection result {threshold_zero_cross_test} of wav {key}.')
            end_point_dic[key] = threshold_zero_cross_test
        return end_point_dic


def main():
    wave_data = energy_zero_crossing_rate_extract.read_wav_from_file()
    energy_zero = energy_zero_crossing_rate_extract.EnergyZeroCrossingRateExtract(wave_data)
    energy_dic = energy_zero.energy_main()
    zero_dic = energy_zero.zero_main()
    end_point = EndPointDetect(energy_dic, zero_dic)
    end_point_dic = end_point.end_point_detect()
    energy_zero_crossing_rate_extract.write_end_point_file(wave_data, end_point_dic, pcm_file)


if __name__ == '__main__':
    main()
