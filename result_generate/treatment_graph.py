import matplotlib.pyplot as plt
import seaborn as sns
from default_config import fig_save_folder
from default_config import treatment_result_inference_folder
import numpy as np
import os
import csv
import pandas as pd


sns.set()
plt.rc('font', size=6)          # controls default text sizes
plt.rc('axes', titlesize=6)     # fontsize of the axes title
plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=6)  # fontsize of the figure title


color_pallete = sns.color_palette(palette='Accent')


def read_data(data_name, sample_id, feature_list):
    data, time = read_treatment_data(data_name, feature_list)
    plot_data = data[sample_id]
    model_name_list = []
    for key in plot_data:
        if not ('origin' in key):
            model_name_list.append(key)
    return plot_data, model_name_list


def main():
    sample_id = 'test_1'
    hao_feature_list = ['a', 'tau_p', 'n', 'c']
    zheng_feature_list = ['a', 'tau', 'n', 'c']
    hao_data, hao_model_list = \
        read_data('hao_true_lmci,True,n,52,0,accept.csv', sample_id, hao_feature_list)
    # zheng_data, zheng_model_list = \
    #     read_data('zheng,False,n,0,0.csv', sample_id, zheng_feature_list)

    hao_model_list = sorted(hao_model_list)
    # zheng_model_list = sorted(zheng_model_list)
    # assert len(hao_model_list) == len(zheng_model_list)
    # for i in range(len(hao_model_list)):
    #     assert hao_model_list[i] == zheng_model_list[i]
    model_name_list = hao_model_list

    # model_name_list = ['oracle_treatment', 'model_2_treatment']
    print('order of model names: {}'.format(model_name_list))


    fig, axs = plt.subplots(1, 4, figsize=[7.5, 2], dpi=300)
    hao_start_time = 50
    hao_treat_time = 52
    hao_end_time = 56
    hao_time_list = [0.05 * i * (hao_end_time - hao_start_time) + hao_start_time + 0.4 for i in range(20)]
    for i in range(4):
        hao_feature = hao_feature_list[i]
        for j, model_name in enumerate(model_name_list):
            hao_value = hao_data[model_name][hao_feature]['mean']
            if j == 0:
                color_idx = 1
            elif j == 1:
                color_idx = 6
            else:
                raise ValueError('')
            if 'oracle' not in model_name:
                hao_max = hao_data[model_name][hao_feature]['max']
                hao_min = hao_data[model_name][hao_feature]['min']
                axs[i].fill_between(hao_time_list, hao_min, hao_max, color=color_pallete[color_idx], alpha=0.2)

            axs[i].plot(hao_time_list, hao_value, label=model_name, color=color_pallete[color_idx])

        axs[i].set_title('Hao ' + hao_feature)
        axs[i].tick_params(axis='both', which='major', pad=0)
        axs[i].set_xlim(hao_start_time, hao_end_time)
        axs[i].set_xticks([hao_start_time, hao_treat_time, hao_end_time])
        axs[i].grid(color='red', linestyle='--', linewidth=0.5)
        axs[i].set_xlabel('time')
        axs[i].set_xlabel('age')
    #

    axs[0].set_ylabel('normalized value')
    # axs[0][0].legend(bbox_to_anchor=(4, 1.4), ncol=7)
    axs[0].set_ylim(-3, 1)
    axs[0].set_yticks([-3, 1])
    axs[1].set_ylim(-2, 0)
    axs[1].set_yticks([-2, 0])
    axs[2].set_ylim(-5, 0)
    axs[2].set_yticks([-5, 0])
    axs[3].set_ylim(-3, 3)
    axs[3].set_yticks([-3, 3])
    axs[0].legend(loc='upper left', ncols=3)
    axs[1].legend(loc='upper left', ncols=3)
    axs[2].legend(loc='upper left', ncols=3)
    axs[3].legend(loc='upper left', ncols=3)

    # handles, labels = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    # plt.figlegend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 1.05), ncol=7, labelspacing=0.2)
    plt.show()
    fig.savefig(os.path.join(fig_save_folder, 'treatment.{}.svg').format(sample_id), bbox_inches='tight')



def read_treatment_data(data_name, feature_list):
    data_path = os.path.join(treatment_result_inference_folder, data_name)
    time_list = []
    data_dict = dict()
    with open(data_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            if i == 0:
                for j in range(4, len(line)):
                    time_list.append(float(line[j]))
                continue
            sample_id, model, feature, data_type = line[0: 4]
            data = np.array([float(item) for item in line[4:]])
            if sample_id not in data_dict:
                data_dict[sample_id] = dict()
            if model not in data_dict[sample_id]:
                data_dict[sample_id][model] = dict()
            if feature not in data_dict[sample_id][model] and feature in feature_list:
                data_dict[sample_id][model][feature] = dict()
            data_dict[sample_id][model][feature][data_type] = data
    return data_dict, time_list


if __name__ == "__main__":
    main()
