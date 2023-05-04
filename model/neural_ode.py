from torch import FloatTensor, chunk, squeeze, stack, transpose, mean, no_grad, LongTensor
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
        self.init_network = LSTM(input_size=input_size*2+1, hidden_size=hidden_size, batch_first=batch_first,
                                 bidirectional=True)
        self.projection_net = Sequential(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, input_size))
        self.derivative = Derivative(input_size, hidden_size)
        self.derivative.apply(self.init_weights)
        self.projection_net.apply(self.init_weights)
        self.init_network.apply(self.init_weights)

    def forward(self, data):
        # data format [batch_size, visit_idx, feature_idx]
        concat_input, _, _, _, label_feature_list, label_time_list, label_mask_list, _ = data
        label_feature, label_mask = FloatTensor(label_feature_list), FloatTensor(label_mask_list)
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
        loss = loss * (1-label_mask)
        loss = loss.mean()
        return predict_value, label_mask_list, label_feature_list, loss

    def predict_init_value(self, concat_input):
        length = LongTensor([len(item) for item in concat_input])
        concat_input = [FloatTensor(item) for item in concat_input]
        data = pad_sequence(concat_input, batch_first=True, padding_value=0).float()
        # packed_data = pack_padded_sequence(data, length, batch_first=True)
        init_seq, _ = self.init_network(data)

        # 根据网上的相关资料，forward pass的index是前一半；这个slice的策略尽管我没查到，但是在numpy上试了试，似乎是对的
        forward = init_seq[range(len(length)), length-1, :self.hidden_size]
        backward = init_seq[:, 0, self.hidden_size:]
        hidden_init = (forward + backward) / 2
        value_init = self.projection_net(hidden_init)
        return value_init

    @staticmethod
    def init_weights(m):
        if isinstance(m, Linear):
            xavier_uniform_(m.weight)


def train(train_dataloader, val_loader, max_epoch, max_iteration, model, optimizer):
    iter_idx = 0
    evaluation(iter_idx, 0, model, train_dataloader, val_loader)
    for epoch_idx in range(max_epoch):
        for batch in train_dataloader:
            iter_idx += 1
            if iter_idx > max_iteration:
                break
            predict, label_mask, label_feature, loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        evaluation(iter_idx, epoch_idx, model, train_dataloader, val_loader)
    return model


def evaluation(iter_idx, epoch_idx, model, train_loader, val_loader):
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
    logger.info('iter: {:>4d}, epoch: {:>4d}, train loss: {:>8.4f}, val loss: {:>8.4f}'
                .format(iter_idx, epoch_idx, train_loss, val_loss))


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
    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    time_offset = 50

    model = NODE(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, time_offset=time_offset)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    _ = train(train_dataloader, validation_dataloader, max_epoch, max_iteration, model,
              optimizer)
    print('')


if __name__ == '__main__':
    framework(args)
