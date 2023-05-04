from default_config import args, logger
import pickle
import numpy as np
from torch import no_grad, mean, FloatTensor
from data_preprocess.data_loader import SequentialVisitDataloader, SequentialVisitDataset, RandomSampler
from model.causal_trajectory_prediction import CausalTrajectoryPrediction
from torch.optim import Adam
from util import get_data_loader


def unit_test(argument):
    batch_first = True if argument['batch_first'] == 'True' else False
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
    if graph_type == 'DAG':
        data = pickle.load(open(data_path, 'rb'))['data']['train']
    elif graph_type == 'ADMG':
        data = pickle.load(open(data_path, 'rb'))['data']['train']
    else:
        raise ValueError('')

    model = CausalTrajectoryPrediction(graph_type=graph_type, constraint=constraint, input_size=input_size,
                                       hidden_size=hidden_size, batch_first=batch_first, mediate_size=mediate_size)
    dataset = SequentialVisitDataset(data)
    sampler = RandomSampler(dataset)
    dataloader = SequentialVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                           minimum_observation=minimum_observation,
                                           reconstruct_input=reconstruct_input, predict_label=predict_label)
    optimizer = Adam(model.parameters())

    for batch in dataloader:
        predict, label_mask, label_feature, loss = model(batch)
        constraint = model.constraint()
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
        assert iter_idx > 0
        if iter_idx == 1 or (iter_idx > 1 and iter_idx % self.update_window == 0):
            with no_grad():
                loss_list = []
                for batch in self.data_loader:
                    predict, label_mask, label_feature, loss = model(batch)
                    loss_list.append(loss)
                loss_mean = mean(loss)
                constraint = model.constraint()
                final_loss = loss_mean + self.current_lambda * constraint + 1 / 2 * self.current_mu * constraint ** 2
                self.val_loss_list.append([iter_idx, final_loss])

        if iter_idx >= 2 * self.update_window and iter_idx % self.update_window == 0:
            assert len(self.val_loss_list) >= 3
            # 按照设计，t在正常情况下应当是在下降的，因此delta按照道理应该是个负数
            t0, t_half, t1 = self.val_loss_list[-3][1], self.val_loss_list[-2][1], self.val_loss_list[-1][1]
            if not (min(t0, t1) < t_half < max(t0, t1)):
                delta_lambda = -np.inf
            else:
                delta_lambda = (t1 - t0) / self.update_window

            if abs(delta_lambda) < self.converge_threshold or delta_lambda > 0:
                with no_grad():
                    constraint = model.constraint()
                    self.constraint_list.append([iter_idx, constraint])
                self.current_lambda += self.current_mu * constraint.item()
                logger.info("Updated lambda to {}".format(self.current_lambda))

                if len(self.constraint_list) >= 2:
                    if self.constraint_list[-1] > self.constraint_list[-2] * self.gamma:
                        self.current_mu *= 10
                        logger.info("Updated mu to {}".format(self.current_mu))
        return self.current_lambda, self.current_mu


def evaluation(epoch_idx, model, train_loader, val_loader):
    with no_grad():
        train_loss_list, val_loss_list = [], []
        for batch in train_loader:
            _, __, ___, loss = model(batch)
            train_loss_list.append(loss)
        for batch in val_loader:
            _, __, ___, loss = model(batch)
            val_loss_list.append(loss)
        train_loss = mean(FloatTensor(train_loss_list)).item()
        val_loss = mean(FloatTensor(val_loss_list)).item()
        constraint = model.constraint().item()
    logger.info('epoch: {:>4d}, train loss: {:>8.4f}, val loss: {:>8.4f}, constraint: {:>8.4f}'
                .format(epoch_idx, train_loss, val_loss, constraint))


def train(train_dataloader, val_loader, max_epoch, max_iteration, model, multiplier_updater, optimizer, threshold):
    iter_idx = 0
    for epoch_idx in range(max_epoch):
        for batch in train_dataloader:
            iter_idx += 1
            if iter_idx > max_iteration:
                break
            lamb, mu = multiplier_updater.update(model, iter_idx)
            constraint = model.constraint()

            if constraint < threshold:
                return model

            predict, label_mask, label_feature, loss = model(batch)

            loss = loss - lamb * constraint - 1 / 2 * mu * constraint ** 2
            loss.backward()
            optimizer.step()
        evaluation(epoch_idx, model, train_dataloader, val_loader)
    return model


def framework(argument):
    # data setting
    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    input_size = argument['input_size']
    mask_tag = argument['mask_tag']
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

    # training
    max_epoch = argument['hidden_size']
    max_iteration = argument['max_iteration']
    batch_size = argument['batch_size']
    stop_threshold = argument['model_converge_threshold']

    # lagrangian
    init_lambda = argument['init_lambda']
    init_mu = argument['init_mu']
    eta = argument['eta']
    gamma = argument['gamma']
    lagrangian_converge_threshold = argument['lagrangian_converge_threshold']
    update_window = argument['update_window']

    dataloader_dict = get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                                      reconstruct_input, predict_label)
    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    model = CausalTrajectoryPrediction(graph_type=graph_type, constraint=constraint, input_size=input_size,
                                       hidden_size=hidden_size, batch_first=batch_first, mediate_size=mediate_size)
    multiplier_updater = LagrangianMultiplierStateUpdater(
        init_lambda=init_lambda, init_mu=init_mu, gamma=gamma, eta=eta, update_window=update_window,
        dataloader=validation_dataloader, converge_threshold=lagrangian_converge_threshold)
    optimizer = Adam(model.parameters())

    _ = train(train_dataloader, validation_dataloader, max_epoch, max_iteration, model, multiplier_updater,
              optimizer, stop_threshold)
    print('')


if __name__ == '__main__':
    framework(args)

