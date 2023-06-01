import numpy as np
import torch
from torch import FloatTensor, chunk, squeeze, unsqueeze, no_grad, LongTensor
from torch.nn import Module, Sequential, Linear, ReLU, LSTM, MSELoss
from torchdiffeq import odeint_adjoint as odeint
from util import get_data_loader
from default_config import logger, args
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence


class Derivative(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.model = Sequential(Linear(input_size, hidden_size), ReLU(), Linear(hidden_size, input_size))

    def forward(self, _, inputs):
        return self.model(inputs)


class NODE(Module):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool, time_offset: float, init_pooling: str):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_offset = time_offset

        # init value estimate module
        self.init_network = LSTM(input_size=input_size*2+1, hidden_size=hidden_size, batch_first=batch_first)
        self.projection_net = \
            Sequential(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, input_size))
        self.derivative = Derivative(input_size, hidden_size)
        self.init_pooling = init_pooling
        # 这里使用MSE应该没啥问题
        # https://ai.stackexchange.com/questions/27341/in-variational-autoencoders-why-do-people-use-mse-for-the-loss
        self.loss = MSELoss(reduction='none')

    def forward(self, data):
        # data format [batch_size, visit_idx, feature_idx]
        concat_input, _, _, _, label_feature_list, \
            label_time_list, label_mask_list, type_list, input_len_list, label_len_list = data

        concat_input = pad_sequence(concat_input, batch_first=True, padding_value=0).float()

        # estimate the init value
        init_value = self.predict_init_value(concat_input, input_len_list, self.init_pooling)

        # length inconsistent case
        init_value_list = chunk(init_value, init_value.shape[0], dim=0)
        loss_sum = 0
        idx = 0
        predict_value_list, loss_list, recons_loss_list, predict_loss_list = [], [], [], []
        for init_value, time, label, mask, sample_type, label_len in \
                zip(init_value_list, label_time_list, label_feature_list, label_mask_list, type_list, label_len_list):
            idx += 1
            time = unsqueeze(squeeze(FloatTensor(time) - self.time_offset), 0)[0]
            predict_value = odeint(self.derivative, init_value, time)
            predict_value = squeeze(predict_value)
            predict_value_list.append(predict_value.detach())
            recons_len, pred_len = torch.sum(sample_type == 1), torch.sum(sample_type == 2)

            loss = self.loss(predict_value, FloatTensor(label)) * (1-FloatTensor(mask))
            loss_mean = loss.sum() / (recons_len+pred_len) / self.input_size
            loss_sum += loss_mean

            loss_new = loss.detach()
            sample_type = unsqueeze(sample_type, dim=1)
            recons_loss = loss_new * (sample_type == 1)
            pred_loss = loss_new * (sample_type == 2)
            recons_loss = torch.sum(recons_loss) / recons_len / self.input_size
            pred_loss = torch.sum(pred_loss) / pred_len / self.input_size
            recons_loss_list.append(recons_loss)
            predict_loss_list.append(pred_loss)

        loss = loss_sum / len(init_value_list)
        recons_loss = torch.mean(FloatTensor(recons_loss_list))
        pred_loss = torch.mean(FloatTensor(predict_loss_list))
        return predict_value_list, label_feature_list, loss, recons_loss, pred_loss

    def predict_init_value(self, data, length_list, pooling):
        # packed_data = pack_padded_sequence(data, length, batch_first=True)
        init_seq, _ = self.init_network(data)
        if pooling == 'mean':
            mask_mat = np.zeros([data.shape[0], data.shape[1], 1])
            for idx in range(len(length_list)):
                mask_mat[idx, :length_list[idx], 0] = 1
            mask_mat = FloatTensor(mask_mat)
            init = init_seq * mask_mat
            init = torch.sum(init, dim=1)
            init = (init / FloatTensor(np.array(length_list)[:, np.newaxis])).float()
        elif pooling == 'none':
            init = init_seq[torch.arange(len(length_list)), LongTensor(length_list)-1, :]
        else:
            raise ValueError('')

        value_init = self.projection_net(init)
        return value_init


def train(train_dataloader, val_loader, max_epoch, max_iteration, model, optimizer, eval_iter_interval,
          eval_epoch_interval):
    iter_idx = 0
    evaluation(iter_idx, 0, model, train_dataloader, val_loader)
    for epoch_idx in range(max_epoch):
        for batch in train_dataloader:
            iter_idx += 1
            if iter_idx > max_iteration:
                break
            predict_value_list, label_feature_list, loss, recons_loss, pred_loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
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
            if loader is None:
                continue
            for batch in loader:
                predict_value_list, label_feature_list, loss, recons_loss, pred_loss = model(batch)
                general.append(loss)
                reconstruct.append(recons_loss)
                predict.append(pred_loss)
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
    init_pooling = argument['init_pooling']

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
    model = NODE(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, time_offset=time_offset,
                 init_pooling=init_pooling)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    _ = train(train_dataloader, validation_dataloader, max_epoch, max_iteration, model,
              optimizer, eval_iter_interval, eval_epoch_interval)
    logger.info('complete')


if __name__ == '__main__':
    framework(args)
