from default_config import logger
from data_preprocess.data_loader import SequentialVisitDataloader, SequentialVisitDataset, RandomSampler
import pickle
import os
from torch import save, no_grad, mean, FloatTensor
from datetime import datetime
import numpy as np
from scipy.integrate import solve_ivp


def get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation, reconstruct_input,
                    predict_label, device):
    # to be revised in future for multi dataset
    logger.info('dataset name: {}'.format(dataset_name))
    dataset_dict = {}
    for split in 'train', 'valid', 'test':
        dataset_dict[split] = pickle.load(open(data_path, 'rb'))['data'][split]
    oracle_graph = pickle.load(open(data_path, 'rb'))['oracle_graph']
    dataloader_dict = {}
    for split in 'train', 'valid', 'test':
        dataset = SequentialVisitDataset(dataset_dict[split])
        sampler = RandomSampler(dataset)
        dataloader = SequentialVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                               minimum_observation=minimum_observation, device=device,
                                               reconstruct_input=reconstruct_input, predict_label=predict_label)
        dataloader_dict[split] = dataloader
    name_id_dict = dataloader_dict['train'].dataset.name_id_dict

    name_type_dict = pickle.load(open(data_path, 'rb'))['feature_type_list']
    stat_dict = pickle.load(open(data_path, 'rb'))['stat_dict']
    id_type_list = ['' for _ in range(len(name_id_dict))]
    for key in name_type_dict:
        if key not in name_id_dict:
            continue
        assert name_type_dict[key] == 'c' or name_type_dict[key] == 'd'
        data_type = 'continuous' if name_type_dict[key] == 'c' else 'discrete'
        idx = name_id_dict[key]
        id_type_list[idx] = data_type
    return dataloader_dict, name_id_dict, oracle_graph, id_type_list, stat_dict


def save_model(model, model_name, folder, epoch_idx, iter_idx, argument, phase):
    dataset_name = argument['dataset_name']
    constraint_type = argument['constraint_type']
    hidden_flag = argument['hidden_flag']
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    assert hidden_flag == 'True' or hidden_flag == 'False'

    file_name = '{}.{}.{}.{}.{}.{}.{}.{}'. \
        format(phase, model_name, dataset_name, hidden_flag, constraint_type, now, epoch_idx, iter_idx)

    model_path = os.path.join(folder, file_name+'.model')
    save(model, model_path)
    config_path = os.path.join(folder, file_name+'.config')
    pickle.dump(argument, open(config_path, 'wb'))
    logger.info('model saved at iter idx: {}, file name: {}'.format(iter_idx, file_name))


class LagrangianMultiplierStateUpdater(object):
    def __init__(self, init_lambda, init_mu, gamma, eta, converge_threshold, update_window, dataloader,
                 max_lambda, max_mu):
        self.init_lambda = init_lambda
        self.current_lambda = init_lambda
        self.init_mu = init_mu
        self.current_mu = init_mu
        self.gamma = gamma
        self.eta = eta
        self.max_lambda = max_lambda
        self.max_mu = max_mu
        self.converge_threshold = converge_threshold
        self.update_window = update_window
        self.data_loader = dataloader
        self.val_loss_list = []
        self.constraint_list = []
        # 也没什么道理，就是不想update的太频繁（正常应该在100左右）
        assert update_window >= 5

    def update(self, model, iter_idx):
        with no_grad():
            constraint, _ = model.calculate_constraint()
            assert iter_idx > 0
            if iter_idx == 1 or (iter_idx > 1 and iter_idx % self.update_window == 0):
                # 注意，此处计算loss是在validation dataset上计算。原则上constraint loss重复计算了，但是反正这个计算也不expensive，
                # 重算一遍也没啥影响，出于代码清晰考虑就重算吧
                loss_sum = 0
                for batch in self.data_loader:
                    input_list, label_feature_list, label_time_list = batch[0], batch[4], batch[5]
                    label_mask_list, label_type_list = batch[6], batch[7]
                    predict_value_list = model(input_list, label_time_list)
                    output_dict = model.loss_calculate(predict_value_list, label_feature_list, label_mask_list,
                                                       label_type_list)
                    loss = output_dict['loss']

                    loss_sum += loss

                final_loss = loss_sum + self.current_lambda * constraint + 1 / 2 * self.current_mu * constraint ** 2
                self.val_loss_list.append([iter_idx, final_loss])

            if iter_idx >= 3 * self.update_window and iter_idx % self.update_window == 0:
                assert len(self.val_loss_list) >= 3
                # 按照设计，loss在正常情况下应当是在下降的，因此delta按照道理应该是个负数
                t0, t_half, t1 = self.val_loss_list[-3][1], self.val_loss_list[-2][1], self.val_loss_list[-1][1]
                if not (min(t0, t1) < t_half < max(t0, t1)):
                    delta_lambda = -np.inf
                else:
                    delta_lambda = (t1 - t0) / self.update_window
            else:
                delta_lambda = -np.inf

            if abs(delta_lambda) < self.converge_threshold or delta_lambda > 0:
                self.constraint_list.append([iter_idx, constraint])
                new_lambda = self.current_mu * constraint.item() + self.current_lambda
                if new_lambda < self.max_lambda:
                    self.current_lambda = new_lambda
                    logger.info("Updated lambda to {}".format(self.current_lambda))
                else:
                    self.current_lambda = self.max_lambda
                    logger.info("Updated lambda to {}".format(self.current_lambda))

                if len(self.constraint_list) >= 2:
                    if self.constraint_list[-1][1] > self.constraint_list[-2][1] * self.gamma:
                        new_mu = 10 * self.current_mu
                        if new_mu < self.max_mu:
                            self.current_mu = new_mu
                        else:
                            self.current_mu = self.max_mu
                        logger.info("Updated mu to {}".format(self.current_mu))
        return self.current_lambda, self.current_mu


def predict_performance_evaluation(model, loader, loader_fraction, epoch_idx=None, iter_idx=None):
    with no_grad():
        loss_list = []
        for batch in loader:
            input_list, label_feature_list, label_time_list = batch[0], batch[4], batch[5]
            label_mask_list, label_type_list = batch[6], batch[7]
            predict_value_list = model(input_list, label_time_list)
            output_dict = model.loss_calculate(predict_value_list, label_feature_list, label_mask_list, label_type_list)
            loss = output_dict['loss']
            loss_list.append(loss)
        loss = mean(FloatTensor(loss_list)).item()
        graph_constraint, sparse_constraint = model.calculate_constraint()
        graph_constraint = graph_constraint.item()
        sparse_constraint = sparse_constraint.item()
    if epoch_idx is not None:
        logger.info('epoch: {:>4d}, iter: {:>4d}, {:>6s} loss: {:>8.8f}, graph cons: {:>8.8f}, sparse cons: {:>8.8f}'
            .format(epoch_idx, iter_idx, loader_fraction, loss, graph_constraint, sparse_constraint))
    else:
        logger.info('final {} loss: {:>8.8f}, graph cons: {:>8.8f}, sparse cons: {:>8.8f}'
                    .format(loader_fraction, loss, graph_constraint, sparse_constraint))


def preset_graph_converter(id_dict, graph):
    dag_graph = np.zeros([len(id_dict), len(id_dict)])
    bi_graph = np.zeros([len(id_dict), len(id_dict)])
    for key_1 in graph['dag']:
        for key_2 in graph['dag'][key_1]:
            idx_1, idx_2 = id_dict[key_1], id_dict[key_2]
            dag_graph[idx_1, idx_2] = graph['dag'][key_1][key_2]
    for key_1 in graph['bi']:
        for key_2 in graph['bi'][key_1]:
            idx_1, idx_2 = id_dict[key_1], id_dict[key_2]
            bi_graph[idx_1, idx_2] = graph['bi'][key_1][key_2]
    return {'dag': dag_graph, 'bi': bi_graph}


def get_oracle_model(model_name, para_dict, init_dict, time_offset, stat_dict):
    if 'hao_true' in model_name:
        return OracleHaoModel(True, para_dict, init_dict, time_offset, stat_dict)
    elif 'hao_false' in model_name:
        return OracleHaoModel(False, para_dict, init_dict, time_offset, stat_dict)
    else:
        raise ValueError('')


class OracleHaoModel(object):
    def __init__(self, hidden_type, init_dict, para_dict, time_offset, stat_dict):
        self.hidden_type = hidden_type
        self.treatment_feature = None
        self.treatment_value = None
        self.para_dict = para_dict
        self.init_dict = init_dict
        self.time_offset = time_offset
        self.treatment_time = None
        self.stat_dict = stat_dict
        self.name_id_dict = {'a': 0, 'tau_p': 1, 'tau_o': 2, 'n': 3, 'c': 4}

    def set_treatment(self, treatment_feature, treatment_time, treatment_value):
        self.treatment_time = treatment_time - self.time_offset
        self.treatment_feature = treatment_feature
        self.treatment_value = treatment_value

    def derivative(self, t, y):
        assert isinstance(self.hidden_type, bool)
        para = self.para_dict
        if t > self.treatment_time:
            idx = self.name_id_dict[self.treatment_feature]
            y[idx] = self.treatment_value

        if self.hidden_type:
            derivative = [
                para['lambda_a_beta'] * y[0] * (1 - y[0] / para['k_a_beta']),
                para['lambda_tau'] * y[0] * (1 - y[1] / para['k_tau']),
                para['lambda_tau_o'],
                (para['lambda_ntau_o'] * y[2] + para['lambda_ntau_p'] * y[1]) * (1 - y[3] / para['k_n']),
                (para['lambda_cn'] * y[3] + 0.4 * para['lambda_ctau'] * (y[1] + y[2])) * (1 - y[4] / para['k_c'])
            ]
        else:
            derivative = [
                para['lambda_a_beta'] * y[0] * (1 - y[0] / para['k_a_beta']),
                para['lambda_tau'] * y[0] * (1 - y[1] / para['k_tau']),
                para['lambda_tau_o'],
                (para['lambda_ntau_o'] * y[2] + para['lambda_ntau_p'] * y[1]) * (1 - y[3] / para['k_n']),
                (para['lambda_cn'] * y[3] + para['lambda_ctau'] * y[1]) * (1 - y[4] / para['k_c'])
            ]

        if t > self.treatment_time:
            derivative[self.name_id_dict[self.treatment_feature]] = 0
        return derivative

    def inference(self, time_list):
        init = self.init_dict
        initial_state = [init['a'], init['tau_p'], init['tau_o'], init['n'], init['c']]

        time_list = [item - self.time_offset for item in time_list]
        result_list = []
        for visit_time in time_list:
            t_span = 0, visit_time
            full_result = solve_ivp(self.derivative, t_span, initial_state)
            result = full_result.y[:, -1]
            result_list.append(result)

        ordered_list = self.reorganize(result_list)
        return ordered_list

    def reorganize(self, result_list):
        # 由于数据中对特征的排序是按照字母顺序排的，因此这里要重新排序
        # 包括要拿掉数据中观测不到的数据
        a_list = [(item[0]-self.stat_dict['a'][0])/self.stat_dict['a'][1] for item in result_list]
        tau_p_list = [(item[1]-self.stat_dict['tau_p'][0])/self.stat_dict['tau_p'][1] for item in result_list]
        tau_o_list = [(item[2]-self.stat_dict['tau_o'][0])/self.stat_dict['tau_o'][1] for item in result_list]
        n_list = [(item[3]-self.stat_dict['n'][0])/self.stat_dict['n'][1] for item in result_list]
        c_list = [(item[4]-self.stat_dict['c'][0])/self.stat_dict['c'][1] for item in result_list]

        if self.hidden_type:
            return_list = [
                a_list, c_list, n_list, tau_p_list
            ]
        else:
            return_list = [
                a_list, c_list, n_list, tau_o_list, tau_p_list
            ]
        return np.array(return_list)

