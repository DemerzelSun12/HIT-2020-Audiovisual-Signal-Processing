import numpy as np
from logger import *
from functools import reduce

sys.stdout = Logger(sys.stdout)

distance = lambda x, y: np.sqrt(sum(np.square(x[i] - y[i]) for i in range(len(x))))


class DTW(object):
    def __init__(self, model_dic, test_dic):
        self.model_dic = model_dic
        self.test_dic = test_dic

    @staticmethod
    def calculate_distance_of_two_vector(vector1, vector2):
        if len(vector1) == len(vector2):
            cost = sum(distance(vector1[i], vector2[i]) for i in range(len(vector1)))
            print(f'cost matrix {cost}')
            return cost
        else:
            len_vector1 = len(vector1)
            len_vector2 = len(vector2)
            cost_matrix = np.zeros((len_vector1, len_vector2))
            dist = np.zeros((len_vector1, len_vector2))
            for i in range(len_vector1):
                for j in range(len_vector2):
                    dist[i][j] = distance(vector1[i], vector2[j])
                    # print(f'dist[{i}][{j}]: {dist[i][j]}')
            cost_matrix[0][0] = dist[0][0]
            for i in range(1, len_vector1):
                cost_matrix[i][0] = cost_matrix[i - 1][0] + dist[i][0]
            for j in range(1, len_vector2):
                cost_matrix[0][j] = cost_matrix[0][j - 1] + dist[0][j]

            for i in range(1, len_vector1):
                for j in range(1, len_vector2):
                    cost_matrix[i][j] = min(cost_matrix[i - 1][j] + dist[i][j] * 1,
                                            cost_matrix[i - 1][j - 1] + dist[i][j] * 2,
                                            cost_matrix[i][j - 1] + dist[i][j] * 1)
            print(f'cost matrix {cost_matrix[len_vector1 - 1][len_vector2 - 1]}')
            return cost_matrix[len_vector1 - 1][len_vector2 - 1]

    def dtw_main(self):
        correct_num = np.zeros(len(self.model_dic.keys()))
        for key in self.test_dic.keys():
            print(f'test data {key}')
            category = -1
            dist = float('inf')
            for model_key in self.model_dic.keys():
                print(f'test model {model_key} for {key}')
                calculate_distance = self.calculate_distance_of_two_vector(self.model_dic[model_key],
                                                                           self.test_dic[key])
                if calculate_distance < dist:
                    category = model_key
                    dist = calculate_distance
            print(f'data{key} belongs to type {category}.\n')
            if int(key.split('_')[0]) == int(category):
                correct_num[int(category) - 1] += 1
            else:
                print("------------------------------------------------")
        for i in range(len(correct_num)):
            print(f'test over, the correct rate of {i + 1} is: {correct_num[i] / 10}')
