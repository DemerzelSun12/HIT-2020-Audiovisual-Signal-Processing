import numpy as np


def calculate_snr(code_data, decode_data):
    molecular = np.longdouble(0)
    denominator = np.longdouble(0)
    for i in range(len(code_data)):
        # print(f'before: {code_data[i]}, after: {decode_data[i]}')
        molecular += pow(code_data[i], 2)
        denominator += pow(code_data[i] - decode_data[i], 2)
    snr = 10 * np.log10(molecular / denominator)
    print(f'SNR: {snr}')
    return snr
