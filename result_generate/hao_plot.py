import os.path
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from default_config import treatment_result_inference_folder, fig_save_folder
import pickle

sns.set()
plt.rc('font', size=5)          # controls default text sizes
plt.rc('axes', titlesize=5)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=5)    # fontsize of the tick labels
plt.rc('legend', fontsize=5)    # legend fontsize
plt.rc('figure', titlesize=5)  # fontsize of the figure title
color_pallete = sns.color_palette(palette='Accent')

def main():
    sample_id = 'test_430'
    data_name = 'hao_true_lmci,True,n,52,0,DAG.pkl'
    hidden = True
    start_time = 50
    treat_time = 52
    end_time = 56
    data_set = data_name.split(',')[0]

    if hidden:
        feature_list = 'a', 'tau_p', 'n', 'c'
    else:
        feature_list = 'a', 'tau_p', 'tau_o', 'n', 'c'

    data, time = read_treatment_data(data_name, feature_list)
    plot_data = data[sample_id]

    model_name_list = [key for key in plot_data[feature_list[0]]]
    sorted(model_name_list)
    # model_name_list = ['oracle_treatment', 'model_2_treatment']
    print('order of model names: {}'.format(model_name_list))
    treatment_effect_estimation(start_time, end_time, time, data, 'oracle_treatment', 'model_1_treatment')
    figure_plot(start_time, end_time, treat_time, time, plot_data, model_name_list, feature_list, data_set, sample_id)


def treatment_effect_estimation(start_time, end_time, time, full_data, oracle_model, test_model):
    start_idx, end_idx = 1000, -1
    for i, time_point in enumerate(time):
        if time_point >= start_time:
            start_idx = i
            break
    for i, time_point in enumerate(time):
        if time_point <= end_time:
            end_idx = i

    difference_dict = dict()
    for sample_id in full_data:
        difference_dict[sample_id] = dict()
        for feature in full_data[sample_id]:
            if feature not in difference_dict[sample_id]:
                difference_dict[sample_id][feature] = dict()
            oracle_data = full_data[sample_id][feature][oracle_model]
            predict_data = full_data[sample_id][feature][test_model]
            if len(oracle_data.shape) > 1:
                oracle_data = np.mean(oracle_data, axis=0)
            if len(predict_data.shape) > 1:
                predict_data = np.mean(predict_data, axis=0)
            oracle_data = oracle_data[start_idx: end_idx+1]
            predict_data = predict_data[start_idx: end_idx + 1]
            difference_dict[sample_id][feature]['oracle'] = oracle_data
            difference_dict[sample_id][feature]['predict'] = predict_data

    count = 0
    difference_sum = 0
    for sample_id in difference_dict:
        for feature in difference_dict[sample_id]:
            oracle = difference_dict[sample_id][feature]['oracle']
            predict = difference_dict[sample_id][feature]['predict']
            difference = np.mean(np.absolute(oracle-predict))
            difference_sum = difference_sum + difference
            count += 1
    print('RMSE: {}'.format(difference_sum/count))



def figure_plot(start_time, end_time, treat_time, time, plot_data, model_name_list, feature_list, data_set, sample_id):
    for feature in feature_list:
        fig, ax = plt.subplots(figsize=[3, 3], dpi=200, layout='tight')
        idx = 0
        for model_name in model_name_list:
            if 'origin' in model_name:
                continue

            data = plot_data[feature][model_name]
            if len(data.shape) > 1:
                value = np.mean(data, axis=0)
                std = np.std(data, axis=0) * 10
                ax.plot(time, value, label=model_name, color=color_pallete[idx])
                ax.fill_between(time, value - std, value + std, interpolate=True, alpha=0.2, color=color_pallete[idx])
            else:
                value = data
                ax.plot(time, value, label=model_name, color=color_pallete[idx])
            idx += 1
            ax.set_yticks([])
            ax.set_ylim(-6, 6)
            ax.set_xlim(start_time, end_time)
            ax.set_xticks([start_time, treat_time, end_time])
            ax.grid(color = 'red', linestyle = '--', linewidth = 0.25)
            # ax.set_xlabel('time')
        plt.show()
        fig.savefig(os.path.join(fig_save_folder, 'treatment.{}.{}.{}.svg').format(data_set, sample_id, feature))


    # 这个纯粹是为了画图例写的
    fig, ax = plt.subplots(figsize=[3, 3], dpi=200)
    for model_name in model_name_list:
        if 'origin' in model_name:
            continue
        data = plot_data[feature_list[0]][model_name]
        if len(data.shape) > 1:
            value = np.mean(data, axis=0)
        else:
            value = data
        ax.plot(time, value, label=model_name)
        ax.legend()
    plt.show()


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

    new_out = dict()
    for sample_id in output_data:
        new_out[sample_id] = dict()
        for model_name in output_data[sample_id]:
            for feature in output_data[sample_id][model_name]:
                if feature not in new_out[sample_id]:
                    new_out[sample_id][feature] = dict()
                new_out[sample_id][feature][model_name] = output_data[sample_id][model_name][feature]
    return new_out, time


if __name__ == "__main__":
    main()
