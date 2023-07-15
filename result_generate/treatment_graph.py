import matplotlib.pyplot as plt
import os
import seaborn as sns
from default_config import fig_save_folder
from util import read_treatment_data

sns.set()
plt.rc('font', size=6)          # controls default text sizes
plt.rc('axes', titlesize=6)     # fontsize of the axes title
plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=6)  # fontsize of the figure title


color_pallete = sns.color_palette(palette='Accent')


def main():
    # sample_id = 'test_138'
    # data_name = 'hao_true_lmci,True,n,52,0.csv'
    data_name = 'zheng,False,n,0,0.csv'

    if 'hao' in data_name:
        sample_id = 'test_138'
        hidden = True
        data_set = data_name.split(',')[0]
        start_time = 50
        treat_time = 52
        end_time = 56
        if hidden:
            feature_list = ['a', 'tau_p', 'n', 'c']
        else:
            feature_list = ['a', 'tau_p', 'tau_o', 'n', 'c']
    elif 'zheng' in data_name:
        sample_id = 'test_138'
        data_set = data_name.split(',')[0]
        start_time = -10
        treat_time = 0
        end_time = 20
        feature_list = ['a', 'tau', 'n', 'c']
    elif 'auto25' in data_name:
        sample_id = 'test_64'
        data_set = data_name.split(',')[0]
        start_time = 0
        treat_time = 1
        end_time = 2
        feature_list = ['node_i'.format(i) for i in range(5, 25)]
    elif 'auto50' in data_name:
        sample_id = 'test_64'
        data_set = data_name.split(',')[0]
        start_time = 0
        treat_time = 1
        end_time = 2
        feature_list = ['node_i'.format(i) for i in range(5, 50)]
    else:
        raise ValueError('')

    data, time = read_treatment_data(data_name, feature_list)
    plot_data = data[sample_id]

    model_name_list = []
    # for key in plot_data:
    #     if 'treatment' in key:
    #         model_name_list.append(key)
    # model_name_list = [key for key in plot_data]
    # model_name_list = ['NGM_treatment', 'oracle_treatment', 'NODE_treatment', 'CTP_treatment', 'LinearODE_treatment' ]
    # model_name_list = ['oracle', 'CTP', 'NGM', 'NODE', 'LinearODE', 'CF-ODE']
    model_name_list = ['oracle', 'CTP']
    sorted(model_name_list)
    # model_name_list = ['oracle_treatment', 'model_2_treatment']
    print('order of model names: {}'.format(model_name_list))
    # treatment_effect_estimation(start_time, end_time, time, data, 'oracle_treatment', 'model_1_treatment')
    figure_plot(start_time, end_time, treat_time, time, plot_data, model_name_list, feature_list, data_set, sample_id)


def figure_plot(start_time, end_time, treat_time, time, plot_data, model_name_list, feature_list, data_set, sample_id):
    fig, axs = plt.subplots(1, 4, figsize=[8.5, 2], dpi=150)
    for i, feature in enumerate(feature_list):
        for j, model_name in enumerate(model_name_list):
            value = plot_data[model_name][feature]['mean']
            # max_ = plot_data[model_name][feature]['max']
            # min_ = plot_data[model_name][feature]['min']
            axs[i].plot(time, value, label=model_name, color=color_pallete[j])
            # axs[0][i].fill_between(time, min_, max_, interpolate=True, alpha=0.2, color=color_pallete[j])
        axs[i].set_title('Hao model   '+feature)
        # # axs[0][i].legend()
        # axs[0][i].set_ylim(-4, 0)
        # axs[0][i].set_yticks([-4, 0])
        axs[i].tick_params(axis='both', which='major', pad=0)
        axs[i].set_xlim(start_time, end_time)
        axs[i].set_xticks([treat_time])
        axs[i].grid(color = 'red', linestyle = '--', linewidth = 0.25)
            # ax.set_xlabel('time')
        axs[i].set_xlabel('age')

    axs[0].set_ylabel('normalized value')
    axs[2].legend(bbox_to_anchor=(1.5, 1.3), ncol=6)

    axs[0].set_ylim(-3, 0)
    axs[0].set_yticks([-3, 0])
    axs[1].set_ylim(-2, 2)
    axs[1].set_yticks([-2, 2])
    axs[2].set_ylim(-5, 2)
    axs[2].set_yticks([-5, 2])
    axs[3].set_ylim(-4, 3)
    axs[3].set_yticks([-4, 3])

    # axs[1][0].set_yticks([])
    # axs[1][0].set_xticks([])
    # axs[1][1].set_yticks([])
    # axs[1][1].set_xticks([])
    # axs[1][2].set_yticks([])
    # axs[1][2].set_xticks([])
    # axs[1][3].set_yticks([])
    # axs[1][3].set_xticks([])
    #
    # plt.figlegend(line_labels, loc='lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,
    #               prop={'size': 13})  # bbox_to_anchor=(0.5, 0.0), borderaxespad=0.1,

    # plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(fig_save_folder, 'treatment.{}.{}.svg').format(data_set, sample_id), bbox_inches='tight')





if __name__ == "__main__":
    main()
