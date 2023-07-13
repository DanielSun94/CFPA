import os
from default_config import resource_folder, fig_save_folder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

true_dict = {
    'hao_true': np.array([
        [np.e**-1, np.e**-3, np.e**-3, np.e**-1, np.e**-3],
        [np.e**-3, np.e**-1, np.e**-3, np.e**-1, np.e**-3],
        [np.e**-3, np.e**-1, np.e**-1, np.e**-3, np.e**-3],
        [np.e**-3, np.e**-1, np.e**-1, np.e**-1, np.e**-3],
        [np.e**-3, np.e**-1, np.e**-1, np.e**-3, np.e**-3],
    ]),
    'blank': np.array([
        [np.e**-3, np.e**-3, np.e**-3, np.e**-3, np.e**-3],
        [np.e**-3, np.e**-3, np.e**-3, np.e**-3, np.e**-3],
        [np.e**-3, np.e**-3, np.e**-3, np.e**-3, np.e**-3],
        [np.e**-3, np.e**-3, np.e**-3, np.e**-3, np.e**-3],
        [np.e**-3, np.e**-3, np.e**-3, np.e**-3, np.e**-3],
    ]),
}

key_list = ['True Graph', 'CTP', 'NGM', 'CUTS', 'TE-CDE', 'CF-ODE', 'Linear ODE', 'Neural ODE']

def main():
    mat_len = 5
    mat_dict = {}
    data_file_name_dict = {
        'Linear ODE': ['cpa_hao_ht_l_sparse_7.log', 3000],
        'NGM': ['cpa_hao_ht_nl_sparse_0.log', 3000],
        'CTP': ['cpa_hao_ht_nl_DAG_14.log', 3000],
        'Neural ODE': ['cpa_hao_ht_nl_none_3.log', 3000],
        'CF-ODE': ['cpa_hao_ht_nl_none_0.log', 3000],
    }
    for key in data_file_name_dict:
        file_name, iteration_num = data_file_name_dict[key]
        path = os.path.join(resource_folder, 'accept_result', 'log', file_name)
        mat = read_mat(path, iteration_num, mat_len)
        mat_dict[key] = np.log(mat)
    mat_dict["True Graph"] = np.log(true_dict['hao_true'])
    mat_dict["TE-CDE"] = np.log(true_dict['blank'])
    mat_dict["CUTS"] = np.log(true_dict['blank'])
    mat_dict["blank"] = np.log(true_dict['blank'])
    plot_figure(mat_dict)


def read_mat(path, iteration_num, mat_len):
    with open(path, 'r') as f:
        full_data = f.readlines()
        target_idx = 0
        for i, line in enumerate(full_data):
            if 'model saved at iter idx: {}'.format(iteration_num) in line:
                target_idx = i-mat_len
        assert target_idx != 0
        data = full_data[target_idx: target_idx+mat_len]
        data_num = []
        for line in data:
            start_idx, end_idx = line.find('['), line.find(']')
            target_str = line[start_idx+1:  end_idx]
            num_str_list = target_str.split(', ')
            num_list = []
            for num in num_str_list:
                num_list.append(float(num.strip()))
            data_num.append(num_list)
    return np.array(data_num)


def plot_figure(mat_dict):
    # heat map
    # heat_map_list = []
    # min_num = np.min(mat)
    # max_num = np.max(mat)
    # normalized_mat = (mat - min_num) / (max_num - min_num)

    fig, axs = plt.subplots(2, len(key_list), layout='constrained', figsize=(21, 6))
    for i in range(len(key_list)):
        key = key_list[i]
        mat = mat_dict[key]
        axs[0][i].set_title(key)
        sns.heatmap(
            mat,
            ax=axs[0][i],
            cmap="YlGnBu",
            vmin=-3,
            vmax=0,
            linewidths=0.1,
            linecolor='black',
            xticklabels=[],
            yticklabels=[],
            # cbar=True if i == len(mat_dict) - 1 else False
            cbar=False
        )
    for i in range(len(key_list)):
        mat = mat_dict['blank']
        sns.heatmap(
            mat,
            ax=axs[1][i],
            cmap="YlGnBu",
            vmin=-3,
            vmax=0,
            linewidths=0.2,
            linecolor='black',
            xticklabels=[],
            yticklabels=[],
            # cbar=True if i == len(mat_dict) - 1 else False
            cbar=False
        )
    axs[0][0].set_ylabel('Hao', fontsize=13)
    axs[1][0].set_ylabel('Zheng', fontsize=13)
    fig.savefig(os.path.join(fig_save_folder, 'causal_discovery.svg'))
    plt.show()


if __name__ == '__main__':
    main()
