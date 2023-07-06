import os
import numpy as np
import torch
from default_config import ckpt_folder, args
from predict_model_train import get_model as get_ctp_model, get_oracle_causal_graph
from predict_model_train import get_data
import seaborn as sns
import matplotlib.pyplot as plt


def main(argument):
    threshold = 0.001
    dataset = argument['dataset_name']
    model_type = 'CTP'
    device = argument['device']
    prior_causal_mask_name = 'hao_true_causal'
    model_name_list = [
        'predict.CTP.hao_true_lmci.True.DAG.20230327064317308620.4.800.model',
        'predict.CTP.hao_true_lmci.True.DAG.20230327064317309236.6.1100.model',
        'predict.CTP.hao_true_lmci.True.DAG.20230327064317576854.3.600.model'
    ]

    mat_list = []
    if model_type == 'CTP':
        dataloader_dict, name_id_dict, _, id_type_list = get_data(argument)
        label = get_oracle_causal_graph(prior_causal_mask_name, name_id_dict)
        for item in model_name_list:
            model_path = os.path.join(ckpt_folder, item)
            state_dict = torch.load(model_path)
            model = get_ctp_model(argument, id_type_list)
            model.load_state_dict(state_dict)
            model.to(device)
            with torch.no_grad():
                net_list = model.derivative.net_list
                predict = model.derivative.calculate_connectivity_mat(net_list, True).detach().to('cpu').numpy()
                mat_list.append([predict, label])
    else:
        raise ValueError('')
    tpl, tnl, fpl, fnl, acc_l, f1l, c_index_l = calculate_performance(mat_list, threshold)
    for name, metric in zip(['acc', 'f1', 'c_index'], [acc_l, f1l, c_index_l]):
        mean = np.mean(metric)
        std = np.std(metric)
        print('{}, {}, {}, {}'.format(model_type, dataset, mean, std))


    # heat map
    heat_map_list = []
    min_num, max_num = 10, -1
    for item in mat_list:
        heat_map_list.append(item[0])
        if np.min(item[0]) < min_num:
            min_num = np.min(item[0])
        if np.max(item[0]) > max_num:
            max_num = np.max(item[0])
    normalized_list = []
    for item in heat_map_list:
        normalized_list.append((item-min_num) / (max_num-min_num))
    normalized_list.insert(0, mat_list[0][1])

    fig, axn = plt.subplots(1, len(normalized_list), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i in range(len(normalized_list)):
        sns.heatmap(normalized_list[i], ax=axn[i],
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    xticklabels=[],
                    yticklabels=[],
                    cbar_ax=None if i else cbar_ax)

    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.show()


def calculate_performance(mat_list, threshold):
    tpl, tnl, fpl, fnl, accl, f1l, c_indexl = [], [], [], [], [], [], []
    for item in mat_list:
        pred, label = item
        tp = np.sum(pred > threshold * label)
        tn = np.sum((1 - pred > threshold) * (1 - label))
        fp = np.sum(pred > threshold * (1 - label))
        fn = np.sum((1 - pred > threshold) * label)
        tpl.append(tp)
        tnl.append(tn)
        fpl.append(fp)
        fnl.append(fn)
        accl.append((tp + tn) / (fp + fn + tp + tn))
        f1l.append(2 * tp / (2 * tp + fp + fn))

        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])
        pair_count, pair_match = 0, 0
        for i in range(len(label)):
            for j in range(len(label)):
                if label[i] != label[j]:
                    pair_count += 1
                    if label[i] == 1 and label[j] == 0:
                        if pred[i] >= pred[j]:
                            pair_match += 1
                    elif label[i] == 0 and label[j] == 1:
                        if pred[i] <= pred[j]:
                            pair_match += 1
                    else:
                        raise ValueError('')
        c_indexl.append(pair_match / pair_count)
    return tpl, tnl, fpl, fnl, accl, f1l, c_indexl





if __name__ == '__main__':
    main(args)


