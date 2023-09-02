import os
import numpy as np
from treatment_graph import read_treatment_data

def main():
    dataset = 'auto25'
    if dataset == 'hao_true_lmci':
        file_name = 'hao_true_lmci,True,n,52,0.csv'
        feature_list = ['a', 'tau_p', 'n', 'c']
        treatment_time = 52
        treatment_feature = 'n'
    elif dataset == 'zheng':
        file_name = 'zheng,False,n,0,0.csv'
        feature_list = ['a', 'tau', 'n', 'c']
        treatment_time = 0
        treatment_feature = 'n'
    elif dataset == 'auto25':
        file_name = 'auto25,False,node_15,1,1.csv'
        feature_list = ['node_{}'.format(i) for i in range(5, 25)]
        treatment_time = 1
        treatment_feature = 'node_15'
    elif dataset == 'auto50':
        file_name = 'hao_true_lmci,True,n,52,0.csv'
        feature_list = ['node_{}'.format(i) for i in range(5, 50)]
        treatment_time = 1
        treatment_feature = 'node_15'
    else:
        raise ValueError('')
    data_dict, time_list = read_treatment_data(file_name, feature_list)
    valid_idx = 0
    for i, time in enumerate(time_list):
        if treatment_time > time:
            valid_idx = i

    result_dict = dict()
    for sample in data_dict:
        for model_name in data_dict[sample]:
            if model_name not in result_dict:
                result_dict[model_name] = dict()
            for feature in data_dict[sample][model_name]:
                if feature not in result_dict[model_name]:
                    result_dict[model_name][feature] = []
                max_data = data_dict[sample][model_name][feature]['max'][valid_idx:]
                mean_data = data_dict[sample][model_name][feature]['mean'][valid_idx:]
                min_data = data_dict[sample][model_name][feature]['min'][valid_idx:]
                result_dict[model_name][feature].append([max_data, mean_data, min_data])

    for model_name in result_dict:
        full_obs = 0
        correct_obs = 0
        for feature in result_dict[model_name]:
            if feature == treatment_feature:
                continue
            oracle_data = np.array([item[1]for item in result_dict['oracle'][feature]])
            predict_max = np.array([item[0]for item in result_dict[model_name][feature]])
            predict_min = np.array([item[2]for item in result_dict[model_name][feature]])

            c1 = oracle_data < predict_max
            c2 = oracle_data > predict_min
            correct = c1 * c2
            full_obs = full_obs + len(correct) * len(correct[0])
            correct_obs = correct_obs + np.sum(correct)
        print('model name: {}, correct ratio： {}'.format(model_name, correct_obs/full_obs))
    print('')



if __name__ == '__main__':
    main()
