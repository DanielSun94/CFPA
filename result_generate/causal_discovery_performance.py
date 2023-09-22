import os
import numpy as np
import torch
from default_config import ckpt_folder, args, update_argument_info
from predict_model_train import get_model as get_ctp_model, get_oracle_causal_graph
from predict_model_train import get_data
import csv


def main(argument, candidate_path=None):
    threshold = 0.3
    model_type = 'CTP'
    device = argument['device']

    if candidate_path is None:
        model_name_list = [
            ['CTP', 'predict.CTP.zheng.False.sparse.20230518093021261096.best.model', 'zheng', False, 'True',
             'use_data'],
        ]
    else:
        with open(candidate_path, 'r', encoding='utf-8-sig', newline='') as f:
            model_name_list = []
            csv_reader = csv.reader(f)
            for line in csv_reader:
                name, path = line[0], line[1]
                if 'hao' in name.lower():
                    hidden_flag = True
                else:
                    hidden_flag = False

                if 'LODE' in name:
                    non_linear = 'False'
                else:
                    non_linear = "True"
                experiment_name = name.lower()
                if 'mm25' in experiment_name:
                    dataset = "auto25"
                elif 'mm50' in experiment_name:
                    dataset = 'auto50'
                elif 'hao' in experiment_name:
                    dataset = 'hao_true_lmci'
                else:
                    assert 'zheng' in experiment_name
                    dataset = 'zheng'

                model_name_list.append([name, path, dataset, hidden_flag, non_linear, 'use_data'])

    for model_info in model_name_list:
        model_name, item, dataset, hidden_flag, non_linear, custom_name = model_info
        mat_list = []
        if model_type == 'CTP':
            argument['dataset_name'] = dataset
            argument['non_linear_mode'] = non_linear
            argument['hidden_flag'] = "True" if hidden_flag else "False"
            argument = update_argument_info(argument)
            dataloader_dict, name_id_dict, oracle, id_type_list = get_data(argument)
            label = get_oracle_causal_graph(name_id_dict, hidden_flag, custom_name, oracle)

            model_path = os.path.join(ckpt_folder, item)
            state_dict = torch.load(model_path, map_location='cuda:0')
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
        for name, metric in zip(['c_index'], [acc_l, f1l, c_index_l]):
            mean = np.mean(metric)
            std = np.std(metric)
            print('{}, {}, {}, {}, {}'.format(model_name, dataset, name, mean, std))


def calculate_performance(mat_list, threshold):
    assert len(mat_list) == 1
    pred, label = mat_list[0]

    b_tp, b_tn, b_fp, b_fn, b_acc, b_f1 = -1, -1, -1, -1, -1, -1
    for i in range(1000):
        cut = threshold / 1000 * (1+i)
        tp = np.sum(pred > cut * label)
        tn = np.sum((1 - pred > cut) * (1 - label))
        fp = np.sum(pred > cut * (1 - label))
        fn = np.sum((1 - pred > cut) * label)
        acc = (tp + tn) / (fp + fn + tp + tn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > b_f1:
            b_f1 = f1
            b_acc = acc
            b_fn = fn
            b_fp = fp
            b_tn = tn
            b_tp = tp

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

    return [b_tp], [b_tn], [b_fp], [b_fn], [b_acc], [b_f1], [pair_match / pair_count]





if __name__ == '__main__':
    path = os.path.abspath('../candidate.csv')
    main(args, path)


