from default_config import logger
from data_preprocess.data_loader import SequentialVisitDataloader, SequentialVisitDataset, RandomSampler
import pickle
import os
from torch import save, no_grad, mean, FloatTensor
from datetime import datetime
import numpy as np


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

    name_type_dict = pickle.load(open(data_path, 'rb'))['type_list']
    id_type_list = ['' for _ in range(len(name_id_dict))]
    for key in name_type_dict:
        if key not in name_id_dict:
            continue
        assert name_type_dict[key] == 'c' or name_type_dict[key] == 'd'
        data_type = 'continuous' if name_type_dict[key] == 'c' else 'discrete'
        idx = name_id_dict[key]
        id_type_list[idx] = data_type
    return dataloader_dict, name_id_dict, oracle_graph, id_type_list


def save_model(model, model_name, folder, epoch_idx, iter_idx, argument, phase):
    dataset_name = argument['dataset_name']
    graph_type = argument['graph_type']
    constraint_type = argument['constraint_type']
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = '{}.{}.{}.{}.{}.{}.{}.{}'.\
        format(phase, model_name, dataset_name, graph_type, constraint_type, now, epoch_idx, iter_idx)
    model_path = os.path.join(folder, file_name+'.model')
    save(model, model_path)
    config_path = os.path.join(folder, file_name+'.config')
    pickle.dump(argument, open(config_path, 'wb'))
    logger.info('model saved at iter idx: {}, file name: {}'.format(iter_idx, file_name))


class LagrangianMultiplierStateUpdater(object):
    def __init__(self, init_lambda, init_mu, gamma, eta, converge_threshold, update_window, dataloader):
        self.init_lambda = init_lambda
        self.current_lambda = init_lambda
        self.init_mu = init_mu
        self.current_mu = init_mu
        self.gamma = gamma
        self.eta = eta
        self.converge_threshold = converge_threshold
        self.update_window = update_window
        self.data_loader = dataloader
        self.val_loss_list = []
        self.constraint_list = []
        # 也没什么道理，就是不想update的太频繁（正常应该在100左右）
        assert update_window >= 5

    def update(self, model, iter_idx):
        with no_grad():
            constraint = model.calculate_constraint()
            assert iter_idx > 0
            if iter_idx == 1 or (iter_idx > 1 and iter_idx % self.update_window == 0):
                # 注意，此处计算loss是在validation dataset上计算。原则上constraint loss重复计算了，但是反正这个计算也不expensive，
                # 重算一遍也没啥影响，出于代码清晰考虑就重算吧
                loss_sum = 0
                for batch in self.data_loader:
                    input_list, _, _, _, label_feature_list, label_time_list, label_mask_list, label_type_list,\
                        _, _, _ = batch
                    predict_value_list = model(input_list, label_time_list)
                    output_dict = model.loss_calculate(predict_value_list, label_feature_list, label_mask_list,
                                                       label_type_list)
                    loss = output_dict['loss']

                    loss_sum += loss

                final_loss = loss_sum + self.current_lambda * constraint + 1 / 2 * self.current_mu * constraint ** 2
                self.val_loss_list.append([iter_idx, final_loss])

            if iter_idx >= 2 * self.update_window and iter_idx % self.update_window == 0:
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
                self.current_lambda += self.current_mu * constraint.item()
                logger.info("Updated lambda to {}".format(self.current_lambda))

                if len(self.constraint_list) >= 2:
                    if self.constraint_list[-1][1] > self.constraint_list[-2][1] * self.gamma:
                        self.current_mu *= 10
                        logger.info("Updated mu to {}".format(self.current_mu))
        return self.current_lambda, self.current_mu


def predict_performance_evaluation(model, loader, loader_fraction, epoch_idx=None, iter_idx=None):
    with no_grad():
        loss_list = []
        for batch in loader:
            input_list, _, _, _, label_feature_list, label_time_list, label_mask_list, label_type_list, _, _, _ = batch
            predict_value_list = model(input_list, label_time_list)
            output_dict = model.loss_calculate(predict_value_list, label_feature_list, label_mask_list, label_type_list)
            loss = output_dict['loss']
            loss_list.append(loss)
        loss = mean(FloatTensor(loss_list)).item()
        constraint = model.calculate_constraint().item()
    if epoch_idx is not None:
        logger.info('epoch: {:>4d}, iter: {:>4d}, {:>6s} loss: {:>8.4f}, constraint: {:>8.4f}'
                    .format(epoch_idx, iter_idx, loader_fraction, loss, constraint))
    else:
        logger.info('final {} loss: {:>8.4f}, constraint: {:>8.4f}'.format(loader_fraction, loss, constraint))


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
