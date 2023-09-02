import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from default_config import fig_save_folder

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
    fig, axs = plt.subplots(1, 4, figsize=[7.5, 2], dpi=300)

    model_number = np.array(["2", "4", "8", "16", '32', '64'])
    hao = np.array([0.42, 0.54, 0.64, 0.70, 0.77, 0.78])
    zheng = np.array([0.64, 0.83, 0.78, 0.85, 0.87, 0.86])
    auto25 = np.array([0.13, 0.19, 0.23, 0.27, 0.35, 0.43])
    auto50 = np.array([0.10, 0.16, 0.20, 0.26, 0.34, 0.36])

    axs[0].bar(model_number, hao)
    axs[1].bar(model_number, zheng)
    axs[2].bar(model_number, auto25)
    axs[3].bar(model_number, auto50)

    axs[0].set_title('Hao')
    axs[1].set_title('Zheng')
    axs[2].set_title('MM-25')
    axs[3].set_title('MM-50')

    axs[0].set_ylabel('inclusion rate')
    axs[0].set_xlabel('model number')
    axs[1].set_xlabel('model number')
    axs[2].set_xlabel('model number')
    axs[3].set_xlabel('model number')
    axs[0].set_ylim([0.2, 0.8])
    axs[1].set_ylim([0.6, 0.9])
    axs[1].set_ylim([0.0, 0.6])
    axs[1].set_ylim([0.0, 0.45])
    axs[0].set_yticks([0.2, 0.4, 0.6, 0.8])
    axs[1].set_yticks([0.6, 0.7, 0.8, 0.9])
    axs[2].set_yticks([0.0, 0.2, 0.4, 0.6])
    axs[3].set_yticks([0.0, 0.15, 0.3, 0.45])
    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(fig_save_folder, 'treatment_inclusion.svg'))

if __name__ == '__main__':
    main()

