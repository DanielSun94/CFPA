import os.path
from datetime import datetime
from default_config import args, logger, ckpt_folder, adjacency_mat_folder
import pickle
import numpy as np
from torch import no_grad, mean, FloatTensor, save
from data_preprocess.data_loader import SequentialVisitDataloader, SequentialVisitDataset, RandomSampler
from model.causal_trajectory_prediction import CausalTrajectoryPrediction
from torch.optim import Adam
from util import get_data_loader


def unit_test(argument):
    batch_first = True if argument['batch_first'] == 'True' else False
    time_offset = argument['time_offset']
    data_path = argument['data_path']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    batch_size = argument['batch_size']
    mediate_size = argument['mediate_size']
    minimum_observation = argument['minimum_observation']
    input_size = argument['input_size']
    mask_tag = argument['mask_tag']
    hidden_size = argument['hidden_size']
    predict_label = True if argument['predict_label'] == 'True' else False
    graph_type = argument['graph_type']
    constraint = argument['constraint_type']
    clamp_edge_threshold = argument['clamp_edge_threshold']
    device = argument['device']
    bidirectional = argument['init_net_bidirectional']
    dataset_name = argument['dataset_name']

    data = pickle.load(open(data_path, 'rb'))['data']['train']

    model = CausalTrajectoryPrediction(graph_type=graph_type, constraint=constraint, input_size=input_size,
                                       hidden_size=hidden_size, batch_first=batch_first, mediate_size=mediate_size,
                                       time_offset=time_offset, clamp_edge_threshold=clamp_edge_threshold,
                                       bidirectional=bidirectional, device=device, dataset_name=dataset_name)
    dataset = SequentialVisitDataset(data)
    sampler = RandomSampler(dataset)
    dataloader = SequentialVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                           minimum_observation=minimum_observation, device=device,
                                           reconstruct_input=reconstruct_input, predict_label=predict_label)
    optimizer = Adam(model.parameters())

    for batch in dataloader:
        _, __, ___, loss, ____, _____ = model(batch)
        constraint = model.calculate_constraint()
        loss = loss - 1 * constraint - 1/2 * constraint**2
        loss.backward()
        optimizer.step()
    logger.info('success')


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
        assert update_window > 5

    def update(self, model, iter_idx):
        with no_grad():
            constraint = model.calculate_constraint()
            assert iter_idx > 0
            if iter_idx == 1 or (iter_idx > 1 and iter_idx % self.update_window == 0):
                # 注意，此处计算loss是在validation dataset上计算。原则上constraint loss重复计算了，但是反正这个计算也不expensive，
                # 重算一遍也没啥影响，出于代码清晰考虑就重算吧
                loss_sum = 0
                for batch in self.data_loader:
                    _, __, ___, loss, ____, _____ = model(batch)
                    loss_sum += loss

                final_loss = loss + self.current_lambda * constraint + 1 / 2 * self.current_mu * constraint ** 2
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
                    if self.constraint_list[-1] > self.constraint_list[-2] * self.gamma:
                        self.current_mu *= 10
                        logger.info("Updated mu to {}".format(self.current_mu))
        return self.current_lambda, self.current_mu


def evaluation(model, loader, loader_fraction, epoch_idx=None, iter_idx=None):
    with no_grad():
        loss_list = []
        for batch in loader:
            _, __, ___, loss, ____, _____ = model(batch)
            loss_list.append(loss)
        loss = mean(FloatTensor(loss_list)).item()
        constraint = model.calculate_constraint().item()
    if epoch_idx is not None:
        logger.info('epoch: {:>4d}, iter: {:>4d}, {:>6s} loss: {:>8.4f}, constraint: {:>8.4f}'
                    .format(epoch_idx, iter_idx, loader_fraction, loss, constraint))
    else:
        logger.info('Final {} loss: {:>8.4f}, val loss: {:>8.4f}, constraint: {:>8.4f}'
                    .format(loader_fraction, loader_fraction, loss, constraint))


def train(train_dataloader, val_loader, model, multiplier_updater, optimizer, argument):
    max_epoch = argument['max_epoch']
    max_iteration = argument['max_iteration']
    model_converge_threshold = argument['model_converge_threshold']
    eval_iter_interval = argument['eval_iter_interval']
    clamp_edge_flag = argument['clamp_edge_flag']
    save_interval = argument['save_iter_interval']
    assert clamp_edge_flag == 'True' or clamp_edge_flag == 'False'
    clamp_edge_flag = True if clamp_edge_flag == 'True' else False

    iter_idx = 0
    evaluation(model, val_loader, 'valid', 0, 0)
    evaluation(model, train_dataloader, 'train', 0, 0)
    model.generate_graph(0, adjacency_mat_folder)
    logger.info('--------------------start training--------------------')
    for epoch_idx in range(max_epoch):
        for batch in train_dataloader:
            iter_idx += 1
            if iter_idx > max_iteration:
                break
            lamb, mu = multiplier_updater.update(model, iter_idx)
            constraint = model.calculate_constraint()

            if constraint < model_converge_threshold:
                return model

            _, __, ___, loss, ____, _____ = model(batch)
            loss = loss + lamb * constraint + 1 / 2 * mu * constraint ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 删除部分边确保稀疏性，前面几次不做clamp，确保不要一开始因为初始化的原因出什么毛病
            if iter_idx > 20 and clamp_edge_flag:
                model.clamp_edge()

            if iter_idx % eval_iter_interval == 0:
                evaluation(model, val_loader, 'valid', epoch_idx, iter_idx)
                evaluation(model, train_dataloader, 'train', epoch_idx, iter_idx)
                model.generate_graph(iter_idx, adjacency_mat_folder)

            if iter_idx % save_interval == 0:
                save_model(model, 'CPA', ckpt_folder, epoch_idx, iter_idx, argument)
    return model


def save_model(model, model_name, folder, epoch_idx, iter_idx, argument):
    dataset_name = argument['dataset_name']
    graph_type = argument['graph_type']
    constraint_type = argument['constraint_type']
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = '{}.{}.{}.{}.{}.{}.{}.model'.\
        format(model_name, dataset_name, graph_type, constraint_type, now, epoch_idx, iter_idx)
    path = os.path.join(folder, file_name)
    save(model, path)
    logger.info('model saved at iter idx: {}'.format(iter_idx))



def framework(argument):
    # data setting
    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    input_size = argument['input_size']
    mask_tag = argument['mask_tag']
    time_offset = argument['time_offset']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False

    # graph setting
    graph_type = argument['graph_type']
    constraint = argument['constraint_type']

    # data loader setting
    batch_first = True if argument['batch_first'] == 'True' else False
    minimum_observation = argument['minimum_observation']

    # model setting
    hidden_size = argument['hidden_size']
    mediate_size = argument['mediate_size']
    bidirectional = argument['init_net_bidirectional']

    # training
    batch_size = argument['batch_size']
    clamp_edge_threshold = argument['clamp_edge_threshold']
    device = argument['device']

    # lagrangian
    init_lambda = argument['init_lambda']
    init_mu = argument['init_mu']
    eta = argument['eta']
    gamma = argument['gamma']
    lagrangian_converge_threshold = argument['lagrangian_converge_threshold']
    update_window = argument['update_window']

    dataloader_dict = get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                                      reconstruct_input, predict_label, device=device)
    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']
    test_dataloader = dataloader_dict['test']

    model = CausalTrajectoryPrediction(graph_type=graph_type, constraint=constraint, input_size=input_size,
                                       hidden_size=hidden_size, batch_first=batch_first, mediate_size=mediate_size,
                                       time_offset=time_offset, clamp_edge_threshold=clamp_edge_threshold,
                                       bidirectional=bidirectional, device=device, dataset_name=dataset_name)
    multiplier_updater = LagrangianMultiplierStateUpdater(
        init_lambda=init_lambda, init_mu=init_mu, gamma=gamma, eta=eta, update_window=update_window,
        dataloader=validation_dataloader, converge_threshold=lagrangian_converge_threshold)
    optimizer = Adam(model.parameters())

    model = train(train_dataloader, validation_dataloader, model, multiplier_updater, optimizer, argument)
    evaluation(model, test_dataloader, 'test')
    print('')


if __name__ == '__main__':
    framework(args)
