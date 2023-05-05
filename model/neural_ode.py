import numpy as np
import torch
from torch import FloatTensor, chunk, squeeze, stack, no_grad, transpose
from torch.nn import Module, Sequential, Linear, ReLU, LSTM, MSELoss
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint
from torch.nn.init import xavier_uniform_
from util import get_data_loader
from default_config import logger, args
from torch.optim import Adam


class Derivative(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.model = Sequential(Linear(input_size, hidden_size), ReLU(), Linear(hidden_size, input_size))

    def forward(self, _, inputs):
        return self.model(inputs)


class NODE(Module):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool, time_offset: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_offset = time_offset

        # init value estimate module
        self.init_network = LSTM(input_size=input_size*2+1, hidden_size=hidden_size, batch_first=batch_first)
        self.projection_net = \
            Sequential(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, input_size))
        self.derivative = Derivative(input_size, hidden_size)
        self.derivative.apply(self.init_weights)
        self.projection_net.apply(self.init_weights)
        self.init_network.apply(self.init_weights)

        self.linear = Linear(4, 5)

    def forward(self, data):
        # data format [batch_size, visit_idx, feature_idx]
        concat_input, _, _, _, label_feature_list, label_time_list, label_mask_list, _ = data
        label_feature = FloatTensor(label_feature_list)
        label_mask_list = FloatTensor(label_mask_list)
        # estimate the init value
        init_value = self.predict_init_value(concat_input)

        # predict label
        label_time_list = FloatTensor(label_time_list)
        init_value_list = chunk(init_value, init_value.shape[0], dim=0)
        time_list = chunk(label_time_list, label_time_list.shape[0], dim=0)
        predict_value = []
        for init_value, time in zip(init_value_list, time_list):
            predict_value.append(odeint(self.derivative, init_value, time-self.time_offset))
        predict_value = squeeze(stack(predict_value))

        # calculate loss
        # 这里使用MSE应该没啥问题
        # https://ai.stackexchange.com/questions/27341/in-variational-autoencoders-why-do-people-use-mse-for-the-loss
        mse_loss = MSELoss(reduction='none')
        loss = mse_loss(predict_value, label_feature)
        loss = loss * (1-label_mask_list)
        return predict_value, label_feature_list, loss

    def predict_init_value(self, concat_input):
        length_list = np.array([len(item) for item in concat_input])
        concat_input = [FloatTensor(item) for item in concat_input]
        data = pad_sequence(concat_input, batch_first=True, padding_value=0).float()
        mask_mat = np.zeros([data.shape[0], data.shape[1], 1])
        for idx in range(len(length_list)):
            mask_mat[idx, :length_list[idx], 0] = 1
        mask_mat = FloatTensor(mask_mat)

        # packed_data = pack_padded_sequence(data, length, batch_first=True)
        init_seq, _ = self.init_network(data)
        init_seq = init_seq * mask_mat
        init_seq = torch.sum(init_seq, dim=1)
        init_seq = (init_seq / FloatTensor(length_list[:, np.newaxis])).float()

        value_init = self.projection_net(init_seq)
        return value_init

    @staticmethod
    def init_weights(m):
        if isinstance(m, Linear):
            xavier_uniform_(m.weight)


def train(train_dataloader, val_loader, max_epoch, max_iteration, model, optimizer, eval_iter_interval,
          eval_epoch_interval):
    iter_idx = 0
    evaluation(iter_idx, 0, model, train_dataloader, val_loader)
    for epoch_idx in range(max_epoch):
        for batch in train_dataloader:
            iter_idx += 1
            if iter_idx > max_iteration:
                break
            predict, label_feature, loss = model(batch)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            if eval_iter_interval > 0 and iter_idx % eval_iter_interval == 0:
                evaluation(iter_idx, epoch_idx, model, train_dataloader, val_loader)
        if eval_epoch_interval > 0 and epoch_idx % eval_epoch_interval == 0:
            evaluation(iter_idx, epoch_idx, model, train_dataloader, val_loader)
    return model


def evaluation(iter_idx, epoch_idx, model, train_loader, val_loader):
    with no_grad():
        t_g_l, t_r_l, t_p_l, v_g_l, v_r_l, v_p_l = [], [], [], [], [], []
        for loader, [general, reconstruct, predict] in \
                zip([train_loader, val_loader], [[t_g_l, t_r_l, t_p_l], [v_g_l, v_r_l, v_p_l]]):
            for batch in loader:
                recons_flag = np.array(batch[7]) == 1
                predict_flag = np.array(batch[7]) == 0
                _, __, loss = model(batch)
                loss = loss.mean(dim=1)
                general.append(loss.mean())
                reconstruct.append((loss * recons_flag).sum() / np.sum(recons_flag))
                predict.append((loss * predict_flag).sum() / np.sum(predict_flag))
    t_g_l = np.mean(t_g_l)
    t_r_l = np.mean(t_r_l)
    t_p_l = np.mean(t_p_l)
    v_g_l = np.mean(v_g_l)
    v_r_l = np.mean(v_r_l)
    v_p_l = np.mean(v_p_l)
    logger.info('iter: {:>4d}, epoch: {:>4d}, loss: train: general: {:>8.4f}, reconstruct: {:>8.4f}, predict: '
                '{:>8.4f}, val: general: {:>8.4f}, reconstruct: {:>8.4f}, predict: {:>8.4f}'
                .format(iter_idx, epoch_idx, t_g_l, t_r_l, t_p_l, v_g_l, v_r_l, v_p_l))


def framework(argument):
    # data setting
    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    input_size = argument['input_size']
    mask_tag = argument['mask_tag']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False

    # data loader setting
    batch_first = True if argument['batch_first'] == 'True' else False
    minimum_observation = argument['minimum_observation']

    # model setting
    hidden_size = argument['hidden_size']

    # training
    max_epoch = argument['max_epoch']
    max_iteration = argument['max_iteration']
    batch_size = argument['batch_size']
    learning_rate = argument['learning_rate']
    dataloader_dict = get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                                      reconstruct_input, predict_label)
    eval_iter_interval = argument['eval_iter_interval']
    eval_epoch_interval = argument['eval_epoch_interval']
    time_offset = argument['time_offset']

    # data loader
    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    # model
    model = NODE(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, time_offset=time_offset)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    _ = train(train_dataloader, validation_dataloader, max_epoch, max_iteration, model,
              optimizer, eval_iter_interval, eval_epoch_interval)
    logger.info('complete')


if __name__ == '__main__':
    framework(args)
