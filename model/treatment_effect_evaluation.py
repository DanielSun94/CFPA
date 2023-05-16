import os
from default_config import ckpt_folder, logger, args as argument
from torch.nn.utils.rnn import pad_sequence
from torch import load, zeros_like, FloatTensor, chunk, cat, randn, LongTensor, unsqueeze, permute, min, max, rand, \
    squeeze
from geomloss import SamplesLoss
from torch.nn import Module, LSTM, Sequential, ReLU, Linear, ParameterList
import numpy as np
from torchdiffeq import odeint_adjoint as odeint


class TreatmentEffectEstimator(Module):
    def __init__(self, model_ckpt_path, dataset_name, treatment_feature, treatment_time, treatment_value, device,
                 name_id_dict, mode, sample_multiplier, batch_size, input_size, oracle_graph=None):
        """
        此处 mode代表了与干预直接关联时，遇到了有confounder时的处理策略
        这里根据oracle graph的不同，其实可能存在五种可能的情况
        1. reciprocal代表特征间之间没有任何confounder，他们是纯粹的互为因果关系
        2. reciprocal with confounder，特征间互为因果，但是同时也的确受第三个confounder的影响
        3. full confounded代表特征间没有互为因果关系，他们是纯粹的被第三个confounder给控制，从而展现出了相关性
        4. forward，代表特征间有因果关系，也有confounder，因果关系的指向是Treatment指向另一个变量
        5. backward，代表特征间有因果关系，也有confounder，因果关系的指向是另一个变量指向Treatment变量
        """

        super().__init__()
        model = load(model_ckpt_path)

        assert isinstance(name_id_dict, dict) and isinstance(treatment_feature, str) and \
               (treatment_feature in name_id_dict)
        # treatment time为-1，-2时表示施加干预的时间是最后一次入院的时间/倒数第二次入院的时间，是正数时代表任意指定时间
        assert treatment_time == -1 or treatment_time == -2 or \
               (treatment_time > 0 and isinstance(treatment_time, float))
        # treatment value -1, -2代表施加干预的值即为最后一次入(倒数第二次)院的指标观测值，正数时代表任意指定值
        assert treatment_value == -1 or treatment_value == -2 \
               or (treatment_value > 0 and isinstance(treatment_value, float))
        assert dataset_name == model.dataset_name
        self.treatment_feature = treatment_feature
        self.treatment_idx = name_id_dict[treatment_feature]
        self.treatment_value = treatment_value
        self.treatment_time = treatment_time
        self.sample_multiplier = sample_multiplier
        self.input_size = input_size
        self.name_id_dict = name_id_dict
        self.dataset_name = dataset_name
        self.mode = mode
        self.batch_size = batch_size
        self.device = device

        # original model不参与参数更新，参数锁定
        model.device = device
        model.causal_derivative.device = device
        self.origin_model = model.to(device).requires_grad_(False)
        self.origin_model_adjacency = self.get_model_adjacency(model, oracle_graph)
        self.confounder_flag = self.get_confounder_flag()

        self.model, self.adjacency = self.rebuild_model(mode)

        self.samples_loss = SamplesLoss('sinkhorn', p=1, blur=0.01, scaling=0.9, backend='tensorized')

    def get_model_adjacency(self, model, oracle_graph):
        adjacency = model.causal_derivative.adjacency
        dag = adjacency['dag']
        bi = adjacency['bi'] if 'bi' in adjacency else zeros_like(dag)
        adjacency = ((dag + bi) > 0).float()
        if oracle_graph is None:
            return adjacency
        else:
            return self.__oracle_graph_reformat(oracle_graph)

    def get_confounder_flag(self):
        confounder_flag = False
        treatment_idx = self.treatment_idx
        for i in range(len(self.origin_model_adjacency)):
            if i == treatment_idx:
                assert self.origin_model_adjacency[i, i] == 0
                continue
            if self.origin_model_adjacency[treatment_idx, i] == self.origin_model_adjacency[i, treatment_idx] and \
                    self.origin_model_adjacency[treatment_idx, i] == 1:
                confounder_flag = True
        return confounder_flag

    def rebuild_model(self, mode):
        # input check
        assert mode in {'full_confounded', 'reciprocal', 'forward', 'backward', 'reciprocal_confounded'}
        original_model = self.origin_model
        original_adjacency = self.origin_model_adjacency
        treatment_idx = self.treatment_idx
        input_size = self.origin_model.input_size
        bidirectional = self.origin_model.init_net_bidirectional
        hidden_size = self.origin_model.hidden_size
        batch_first = self.origin_model.init_network.batch_first
        sample_multiplier = self.sample_multiplier
        device = self.device
        if self.confounder_flag:
            if mode == 'reciprocal':
                model = original_model
                adjacency = original_adjacency
                logger.info('We will use the origin model to conduct treatment analysis as we set mode to "reciprocal"'
                            ' to handle potential confounder'.format(self.treatment_feature))
            elif mode in {'full_confounded', 'forward', 'backward', 'reciprocal_confounded'}:
                model = TrajectoryPrediction(original_adjacency, mode, treatment_idx, input_size, hidden_size,
                                             bidirectional, batch_first, sample_multiplier, device)
                adjacency = model.adjacency
                logger.info('The model requires training before inference as we construct a new model'
                            .format(self.treatment_feature))
            else:
                raise ValueError('')
        else:
            model = original_model
            adjacency = original_adjacency
            logger.info('We will use the origin model to conduct treatment analysis as {} '
                        'is not correlated to a potential confounder'.format(self.treatment_feature))
        return model, adjacency

    def __oracle_graph_reformat(self, oracle_graph):
        name_idx_dict= self.name_id_dict
        adjacency = np.zeros([len(name_idx_dict), len(name_idx_dict)])
        for key_1 in oracle_graph:
            for key_2 in oracle_graph[key_1]:
                value = oracle_graph[key_1][key_2]
                if value > 0:
                    idx_1 = name_idx_dict[key_1]
                    idx_2 = name_idx_dict[key_2]
                    adjacency[idx_1, idx_2] = 1
        adjacency = FloatTensor(adjacency)
        return adjacency


    def re_fit(self, data, time=None):
        # 这里有个点，sinkhorn loss只能在sample内部进行比较
        assert self.confounder_flag and self.mode in\
               {'reciprocal_confounded', 'forward', 'backward', 'full_confounded'}
        # data format [batch_size, visit_idx, feature_idx]
        concat_input_list, _, _, _, label_feature_list, label_time_list, label_mask_list, label_type_list, _, _ = data
        sample_multiplier = self.sample_multiplier
        batch_size = self.batch_size
        input_size = self.input_size
        device = self.device

        original_model = self.origin_model
        model = self.model

        origin_init = original_model.predict_init_value(concat_input_list)
        origin_init_mean, origin_init_std = origin_init[:, :self.input_size], origin_init[:, self.input_size:]
        origin_init_mean, origin_init_std = unsqueeze(origin_init_mean, dim=0), unsqueeze(origin_init_std, dim=0)
        origin_std = randn([sample_multiplier, batch_size, input_size]).to(device)
        origin_init_value = origin_init_mean + origin_std * origin_init_std

        new_init = model.predict_init_value(concat_input_list)
        new_init_mean, new_init_std = new_init[:, :self.input_size+1], new_init[:, self.input_size+1:]
        new_init_mean, new_init_std = unsqueeze(new_init_mean, dim=0), unsqueeze(new_init_std, dim=0)
        new_std = randn([sample_multiplier, batch_size, input_size + 1]).to(device)
        new_init_value = new_init_mean + new_std * new_init_std

        if time is None:
            time = self.get_predict_time(label_time_list)
        else:
            assert isinstance(time, list) and isinstance(time[0], float)
            time = FloatTensor(time)

        origin_init_value = origin_init_value.reshape([self.batch_size * self.sample_multiplier, self.input_size])
        new_init_value = new_init_value.reshape([self.batch_size * self.sample_multiplier, self.input_size+1])

        origin_predict_value = odeint(self.origin_model.causal_derivative, origin_init_value, time)
        new_predict_value = odeint(self.model.derivative, new_init_value, time)

        origin_predict_value, new_predict_value = squeeze(origin_predict_value), squeeze(new_predict_value)
        origin_predict_value = origin_predict_value.reshape([self.sample_multiplier, self.batch_size, self.input_size])
        new_predict_value = new_predict_value.reshape([self.sample_multiplier, self.batch_size, self.input_size+1])
        origin_predict_value = permute(origin_predict_value, (1, 0, 2))
        new_predict_value = permute(new_predict_value, (1, 0, 2))
        # 最后一个数据本质是不可观测的
        new_predict_value = new_predict_value[:, :, :input_size]

        loss_sum = 0
        origin_predict_value_list = chunk(origin_predict_value, chunks=self.batch_size, dim=0)
        new_predict_value_list = chunk(new_predict_value, chunks=self.batch_size, dim=0)
        for origin, new in zip(origin_predict_value_list, new_predict_value_list):
            loss = self.samples_loss(new, origin)
            loss_sum += loss
        return loss_sum

    def get_predict_time(self, label_time_list):
        min_max_list = []
        for item in label_time_list:
            min_max_list.append(
                [float(min(item)), float(max(item))]
            )
        min_max = -10000
        max_min = 10000
        for item in min_max_list:
            if min_max < item[0]:
                min_max = item[0]
            if max_min > item[1]:
                max_min = item[1]
        if min_max >= max_min:
            time = FloatTensor(min_max)
        else:
            time = randn([1]) * (max_min-min_max) + min_max
        time = FloatTensor(time).to(self.device)
        return time


class TrajectoryPrediction(Module):
    """
    这个其实是Causal Trajectory Prediction，但是在这里不需要特别复杂的因果分析设定了，所以结构可以简单一些
    """
    def __init__(self, origin_adjacency, mode, treat_idx, input_size, hidden_size, bidirectional, batch_first,
                 sample_multiplier, device):
        super().__init__()
        self.input_size = input_size
        self.mode = mode
        self.treat_idx = treat_idx
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.sample_multiplier = sample_multiplier
        self.device = device

        self.init_network = LSTM(input_size=input_size*2+1, hidden_size=hidden_size, batch_first=batch_first,
                                 bidirectional=bidirectional)
        self.init_network.to(device)
        self.projection_net = Sequential(
            Linear(hidden_size, hidden_size), ReLU(),
            Linear(hidden_size, (input_size+1) * 2)
        )
        self.projection_net.to(device)
        self.origin_adjacency = origin_adjacency
        self.adjacency = self.set_adjacency()
        self.derivative = Derivative(input_size, hidden_size, self.adjacency, device).to(device)


    def predict_init_value(self, concat_input):
        length = LongTensor([len(item) for item in concat_input]).to(self.device)
        data = pad_sequence(concat_input, batch_first=True, padding_value=0).float()
        init_seq, _ = self.init_network(data)

        if not self.bidirectional:
            hidden_init = init_seq[range(len(length)), length - 1, :]
        else:
            forward = init_seq[range(len(length)), length - 1, :self.hidden_size]
            backward = init_seq[:, 0, self.hidden_size:]
            hidden_init = (forward + backward) / 2
        value_init = self.projection_net(hidden_init)
        return value_init


    def set_adjacency(self):
        origin_adjacency = self.origin_adjacency
        input_size = self.input_size
        treat_idx = self.treat_idx
        mode = self.mode

        adjacency = np.zeros([input_size+1, input_size+1])
        # self-excite of hidden confounder
        adjacency[input_size, input_size] = 1
        for i in range(len(self.origin_adjacency)):
            for j in range(len(self.origin_adjacency[i])):
                adjacency[i, j] = origin_adjacency[i, j]

        confounder_idx_list = []
        for i in range(len(adjacency)):
            if i == treat_idx:
                assert adjacency[i, i] == 0
                continue
            if adjacency[treat_idx, i] == 1 and adjacency[treat_idx, i] == adjacency[i, treat_idx]:
                confounder_idx_list.append(i)
        assert len(confounder_idx_list) > 0

        for idx in confounder_idx_list:
            assert adjacency[treat_idx, idx] == adjacency[idx, treat_idx] and adjacency[idx, treat_idx] == 1
            if mode == 'full_confounded':
                adjacency[treat_idx, idx] = 0
                adjacency[idx, treat_idx] = 0
                adjacency[input_size, treat_idx] = 1
                adjacency[input_size, idx] = 1
            elif mode == 'forward':
                adjacency[idx, treat_idx] = 0
                adjacency[input_size, treat_idx] = 1
                adjacency[input_size, idx] = 1
            elif mode == 'backward':
                adjacency[idx, treat_idx] = 0
                adjacency[input_size, treat_idx] = 1
                adjacency[input_size, idx] = 1
            elif mode == 'reciprocal_confounded':
                adjacency[input_size, treat_idx] = 1
                adjacency[input_size, idx] = 1
            else:
                raise ValueError('')
        return FloatTensor(adjacency)


class Derivative(Module):
    def __init__(self, input_size, hidden_size, adjacency, device):
        super().__init__()
        self.input_size = input_size
        self.adjacency = adjacency
        self.device = device
        self.net_list = ParameterList()
        for i in range(input_size+1):
            net = Sequential(
                Linear(input_size + 1, hidden_size),
                ReLU(),
                Linear(hidden_size, hidden_size),
                ReLU(),
                Linear(hidden_size, 1),
            )
            self.net_list.append(net)

    def forward(self, _, inputs):
        # inputs shape [batch size, input dim]
        # 注意，再次求derivative一定会有一个隐变量，因此假设有input size+1
        assert inputs.shape[1] == self.input_size+1 and len(inputs.shape) == 2
        inputs = unsqueeze(inputs, dim=1)
        inputs = inputs.repeat(1, inputs.shape[2], 1)
        assert inputs.shape[1] == inputs.shape[2] and len(inputs.shape) == 3

        adjacency = unsqueeze(self.adjacency, dim=0).to(self.device)
        inputs = inputs * adjacency
        inputs_list = chunk(inputs, inputs.shape[1], dim=1)

        output_feature = []
        for i in range(self.input_size+1):
            net_i = self.net_list[i]
            input_i = inputs_list[i]
            derivative = net_i(input_i)
            output_feature.append(derivative)
        output_feature = cat(output_feature, dim=2)
        output_feature = squeeze(output_feature, dim=1)
        return output_feature



def main():
    path = 'CPA.hao_false.DAG.default.20230407075131.0.1.model'
    model_path = os.path.join(ckpt_folder, path)


if __name__ == '__main__':
    main()