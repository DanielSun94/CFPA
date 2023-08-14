import os
import numpy as np
import torch
from default_config import ckpt_folder, args
from predict_model_train import get_model as get_ctp_model, get_oracle_causal_graph
from predict_model_train import get_data

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
    threshold = 0.3
    model_type = 'CTP'
    device = argument['device']
    model_name_list = [
        # ['predict.CTP.auto25.False.sparse.20230423122932725636.37.3000.model', 'auto25', False, 'True', 'use_data'],
        # ['predict.CTP.auto25.False.DAG.20230423133012062462.37.3000.model', 'auto25', False, 'True', 'use_data'],
        # ['predict.CTP.auto25.False.sparse.20230423122932750697.37.3000.model', 'auto25', False, 'False', 'use_data'],
        # ['predict.CTP.auto25.False.none.20230412123244647565.37.3000.model', 'auto25', False, 'True', 'use_data'],
        # ['predict.CTP.auto50.False.none.20230412123244389694.37.3000.model', 'auto50', False, 'True', 'use_data'],
        # ['predict.CTP.auto50.False.sparse.20230423122932534001.37.3000.model', 'auto50', False, 'False', 'use_data'],
        # ['predict.CTP.auto50.False.DAG.20230423133005637113.37.3000.model', 'auto50', False, 'True', 'use_data'],
        # ['predict.CTP.auto50.False.sparse.20230423122932549363.37.3000.model', 'auto50', False, 'True', 'use_data'],
        # ['predict.CTP.zheng.False.sparse.20230423080106170328.34.2800.model', 'zheng', False, 'True', 'use_data'],
        # ['predict.CTP.zheng.False.sparse.20230412123244373515.37.3000.model', 'zheng', False, "True", 'use_data'],
        # ['predict.CTP.zheng.False.none.20230412123244238576.37.3000.model', 'zheng', False, "True", 'use_data'],
        # ['predict.CTP.zheng.False.sparse.20230412123244356323.37.3000.model', 'zheng', False, "False", 'use_data']
        # ["predict.CTP.hao_true_lmci.True.DAG.20230410063658996264.37.3000.model", 'hao_true_lmci', True, "True", 'hao_true_causal'],  # DAG 0.86
        # ['predict.CTP.hao_true_lmci.True.sparse.20230410063659278979.37.3000.model', 'hao_true_lmci', True, "False", 'hao_true_causal'],  # Linear 0.72
        # ['predict.CTP.hao_true_lmci.True.none.20230410063659156026.37.3000.model', 'hao_true_lmci', True, "True", 'hao_true_causal'],  # none 0.54
        # ['predict.CTP.hao_true_lmci.True.sparse.20230410063659452740.37.3000.model', 'hao_true_lmci', True, "True", 'hao_true_causal'],  # sparse 0.76
    ]

    for model_info in model_name_list:
        item, dataset, hidden_flag, non_linear, custom_name = model_info
        mat_list = []
        if model_type == 'CTP':
            argument['dataset_name'] = dataset
            argument['non_linear_mode'] = non_linear
            argument['hidden_flag'] = "True" if hidden_flag else "False"
            dataloader_dict, name_id_dict, oracle, id_type_list = get_data(argument)
            label = get_oracle_causal_graph(name_id_dict, hidden_flag, custom_name, oracle)

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
    main(args)


