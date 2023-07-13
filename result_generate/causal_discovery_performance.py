import os
import numpy as np
import torch
from default_config import ckpt_folder, args
from predict_model_train import get_model as get_ctp_model, get_oracle_causal_graph
from predict_model_train import get_data
import seaborn as sns
import matplotlib.pyplot as plt

# Linear ODE
# 'predict.CTP.hao_true_lmci.True.sparse.20230402160017666256.18.3000.model',
# 'predict.CTP.hao_true_lmci.True.sparse.20230402160017675829.18.3000.model'
# NODE
# "predict.CTP.hao_true_lmci.True.none.20230402160017638219.18.3000.model",
# "predict.CTP.hao_true_lmci.True.none.20230402160017622164.18.3000.model",
# "predict.CTP.hao_true_lmci.True.none.20230402160017593823.18.3000.model",
# "predict.CTP.hao_true_lmci.True.none.20230402160017632388.18.3000.model",
# "predict.CTP.hao_true_lmci.True.none.20230402160017551015.18.3000.model",
# "predict.CTP.hao_true_lmci.True.none.20230402160017549878.18.3000.model",
# "predict.CTP.hao_true_lmci.True.none.20230402160017516543.18.3000.model",
# "predict.CTP.hao_true_lmci.True.none.20230402160017555945.18.3000.model"
# NODE Sparse (NGM)
# "predict.CTP.hao_true_lmci.True.sparse.20230402160017628116.18.3000.model",
# "predict.CTP.hao_true_lmci.True.sparse.20230402160017566475.18.3000.model",
# "predict.CTP.hao_true_lmci.True.sparse.20230402160017665773.18.3000.model",
# "predict.CTP.hao_true_lmci.True.sparse.20230402160017561905.18.3000.model",
# "predict.CTP.hao_true_lmci.True.sparse.20230402160017676152.18.3000.model",
# "predict.CTP.hao_true_lmci.True.sparse.20230402160017635273.18.3000.model",
# "predict.CTP.hao_true_lmci.True.sparse.20230402160017673871.18.3000.model",
# "predict.CTP.hao_true_lmci.True.sparse.20230402160017576411.18.3000.model"
#
# CTP
# "predict.CTP.hao_true_lmci.True.DAG.20230402160017432987.18.3000.model",
# "predict.CTP.hao_true_lmci.True.DAG.20230402160017285169.18.3000.model",
# "predict.CTP.hao_true_lmci.True.DAG.20230402160017293297.18.3000.model",
# "predict.CTP.hao_true_lmci.True.DAG.20230402160017319134.18.3000.model",
# "predict.CTP.hao_true_lmci.True.DAG.20230402160017400674.18.3000.model",
# "predict.CTP.hao_true_lmci.True.DAG.20230402160017286170.18.3000.model",
#

def main(argument):
    threshold = 0.005
    dataset = argument['dataset_name']
    model_type = 'CTP'
    device = argument['device']
    causal_mask_name = 'hao_true_causal'
    argument['non_linear_mode'] = "True"
    model_name_list = [
        "predict.CTP.hao_true_lmci.True.DAG.20230402160017432987.18.3000.model",
        "predict.CTP.hao_true_lmci.True.DAG.20230402160017285169.18.3000.model",
        "predict.CTP.hao_true_lmci.True.DAG.20230402160017293297.18.3000.model",
        "predict.CTP.hao_true_lmci.True.DAG.20230402160017319134.18.3000.model",
        "predict.CTP.hao_true_lmci.True.DAG.20230402160017400674.18.3000.model",
        "predict.CTP.hao_true_lmci.True.DAG.20230402160017286170.18.3000.model",
    ]

    mat_list = []
    if model_type == 'CTP':
        dataloader_dict, name_id_dict, _, id_type_list = get_data(argument)
        label = get_oracle_causal_graph(causal_mask_name, name_id_dict)
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
        print('{}, {}, {}, {}, {}'.format(model_type, dataset, name, mean, std))


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


