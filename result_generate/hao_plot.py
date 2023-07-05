import os.path

import numpy as np

from default_config import treatment_result_inference_folder
import pickle
import matplotlib.pyplot as plt


def main():
    time_offset = 50
    treatment_time = 52
    sample_id = 'test_430'
    data_name = 'hao_true_lmci,True,n,52,0,DAG.pkl'
    hidden = True

    if hidden:
        feature_list = 'a', 'tau_p', 'n', 'c'
    else:
        feature_list = 'a', 'tau_p', 'tau_o', 'n', 'c'

    data, time = read_treatment_data(data_name, feature_list)
    plot_data = data[sample_id]
    plot(plot_data, time, feature_list)
    print('')

def plot(plot_data, time, feature_list):
    subplot_num = len(feature_list)
    fig, axs = plt.subplots(1, subplot_num, figsize=(20, 6))

    for model_name in plot_data:
        if 'origin' in model_name:
            continue
        for feature in plot_data[model_name]:
            idx = feature_list.index(feature)
            data = plot_data[model_name][feature]
            if len(data.shape) > 1:
                value = np.mean(data, axis=0)
                std = np.std(data, axis=0)
            else:
                value = data
                std = 0

            # axs[idx].plot(time, value)
            axs[idx].plot(time, value, label=model_name)
            # plt.fill_between(time, value - std, value + std)
            axs[idx].set_xlabel(feature)
            if idx !=0:
                axs[idx].set_yticks([])
    axs[0].legend()
    plt.show()
    print('')


def read_treatment_data(data_name, feature_list):
    data_path = os.path.join(treatment_result_inference_folder, data_name)
    data, time = pickle.load(open(data_path, 'rb'))

    output_data = dict()
    for sample_id in data:
        output_data[sample_id] = dict()
        sample_data = data[sample_id]
        for model_name in sample_data:
            output_data[sample_id][model_name] = dict()
            model_data = sample_data[model_name]
            for feature in feature_list:
                output_data[sample_id][model_name][feature] = model_data[feature]
    return output_data, time


if __name__ == "__main__":
    main()
