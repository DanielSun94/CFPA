import torch
from default_config import logger
from torch.nn.utils.rnn import pad_sequence
from torch import load, FloatTensor, chunk, cat, randn, LongTensor, unsqueeze, permute, min, max, rand, \
    squeeze
from geomloss import SamplesLoss
from torch.nn import Module, LSTM, Sequential, ReLU, Linear, ParameterList
import numpy as np
from torchdiffeq import odeint_adjoint as odeint


class TreatmentEffectEstimator(Module):
    def __init__(self, model_ckpt_path, dataset_name, treatment_idx, treatment_time, treatment_value, device,
                 mode, sample_multiplier, batch_size, input_size, preset_graph, clamp_edge_threshold):
        """
        此处 mode代表了与干预直接关联时，遇到了有confounder时的处理策略
        这里根据oracle graph的不同，其实可能存在三种可能的情况
        1. full confounded代表特征间没有互为因果关系，他们是纯粹的被第三个confounder给控制，从而展现出了相关性
        2. forward，代表特征间有因果关系，也有confounder，因果关系的指向是Treatment指向另一个变量
        3. backward，代表特征间有因果关系，也有confounder，因果关系的指向是另一个变量指向Treatment变量
        """

        super().__init__()
        # 此处model load部分要重写，因为oracle模型的情况也要用这行
        model = load(model_ckpt_path)

        # treatment time为-1，-2时表示施加干预的时间是最后一次入院的时间/倒数第二次入院的时间，是正数时代表任意指定时间
        assert treatment_time == -1 or treatment_time == -2 or \
               (treatment_time > 0 and isinstance(treatment_time, float))
        # treatment value -1, -2代表施加干预的值即为最后一次入(倒数第二次)院的指标观测值，正数时代表任意指定值
        assert treatment_value == -1 or treatment_value == -2 \
               or (treatment_value > 0 and isinstance(treatment_value, float))
        assert dataset_name == model.dataset_name
        self.treatment_idx = treatment_idx
        self.treatment_value = treatment_value
        self.treatment_time = treatment_time
        self.sample_multiplier = sample_multiplier
        self.input_size = input_size
        self.dataset_name = dataset_name
        self.preset_graph = preset_graph
        self.clamp_edge_threshold = clamp_edge_threshold
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.samples_loss = SamplesLoss('sinkhorn', p=1, blur=0.01, scaling=0.9, backend='tensorized')
        self.oracle_model = model.to(device).requires_grad_(False)

        self.graph = preset_graph if preset_graph is not None else \
            self.oracle_model.generate_binary_graph(clamp_edge_threshold)
        self.graph_legal_check(self.graph)
        self.re_fit_flag = self.get_requires_refit_flag()

        # oracle model不参与参数更新，参数锁定
        model.device = device
        model.causal_derivative.device = device

        self.model, self.adjacency = self.build_trajectory_effect_prediction_model(mode)

    def get_requires_refit_flag(self):
        graph = self.graph
        treatment_idx = self.treatment_idx
        bi_adjacency = graph['bi']
        if bi_adjacency[treatment_idx].sum() == 0:
            return False
        else:
            return True

    def graph_legal_check(self, graph):
        error = torch.trace(torch.matrix_exp(FloatTensor(graph['dag']))) - self.input_size
        if 'bi' in graph and graph['bi'] is not None:
            error = torch.sum(torch.matrix_exp(FloatTensor(graph['dag']))  * graph['bi']) + error
        if error > 0:
            logger.warn('graph illegal')

    @staticmethod
    def adjacency_reconstruct(mode, input_size, oracle_graph, treatment_idx):
        assert mode in {'full_confounded', 'forward', 'backward'}
        adjacency_graph = np.zeros([input_size+1, input_size+1])

        graph = oracle_graph['dag']
        if 'bi' in oracle_graph and oracle_graph['bi'] is not None:
            graph = np.logical_or(graph, oracle_graph['bi'])
        for i in range(len(graph)):
            for j in range(len(graph[i])):
                adjacency_graph[i, j] = graph[i, j]

        adjacency_graph = np.array(adjacency_graph, dtype=float)
        for j in range(input_size):
            if adjacency_graph[treatment_idx, j] == adjacency_graph[j, treatment_idx] \
                    and adjacency_graph[treatment_idx, j] == 1:
                if mode == 'full_confounded':
                    adjacency_graph[treatment_idx, j] = 0
                    adjacency_graph[j, treatment_idx] = 0
                    adjacency_graph[input_size, treatment_idx] = 1
                    adjacency_graph[input_size, j] = 1
                elif mode == 'forward':
                    adjacency_graph[j, treatment_idx] = 0
                    adjacency_graph[input_size, treatment_idx] = 1
                    adjacency_graph[input_size, j] = 1
                elif mode == 'backward':
                    adjacency_graph[treatment_idx, j] = 0
                    adjacency_graph[input_size, treatment_idx] = 1
                    adjacency_graph[input_size, j] = 1
                else:
                    raise ValueError('')
        return adjacency_graph

    def build_trajectory_effect_prediction_model(self, mode):
        # build model的任务包含两个，一个是识别treatment的性质（等同于重新构建一个oracle graph）
        # 另一个才是build model
        refit_flag = self.re_fit_flag
        oracle_model = self.oracle_model
        oracle_model.requires_grad_(False)
        if not refit_flag:
            new_model = oracle_model
            return new_model, refit_flag

        device = self.device
        bidirectional = oracle_model.init_net_bidirectional
        hidden_size = oracle_model.hidden_size
        batch_first = oracle_model.init_network.batch_first
        sample_multiplier = self.sample_multiplier
        input_size = self.oracle_model.input_size
        graph = self.graph
        treatment_idx = self.treatment_idx
        adjacency = self.adjacency_reconstruct(mode, input_size, graph, treatment_idx)
        adjacency = FloatTensor(np.eye(input_size+1) + adjacency)

        model = TrajectoryPrediction(adjacency, mode, treatment_idx, input_size, hidden_size,
                                     bidirectional, batch_first, sample_multiplier, device)
        logger.info('The model requires training before inference as we construct a new model')
        return model, adjacency

    def inference(self, data, time):
        concat_input_list, _, _, _, label_feature_list, label_time_list, label_mask_list, label_type_list, _, _ = data
        model = self.model

        new_init = model.predict_init_value(concat_input_list)
        new_init_mean, _ = new_init[:, :self.input_size + 1], new_init[:, self.input_size + 1:]
        time = FloatTensor(time)
        new_predict_value = odeint(self.model.derivative, new_init_mean, time)
        new_predict_value_list = chunk(new_predict_value, chunks=self.batch_size, dim=0)
        return new_predict_value_list


    def re_fit(self, data, time=None):
        # 这里有个点，sinkhorn loss只能在sample内部进行比较
        assert self.mode in {'forward', 'backward', 'full_confounded'}
        # data format [batch_size, visit_idx, feature_idx]
        concat_input_list, _, _, _, label_feature_list, label_time_list, label_mask_list, label_type_list, _, _ = data
        sample_multiplier = self.sample_multiplier
        batch_size = self.batch_size
        input_size = self.input_size
        device = self.device

        oracle_model = self.oracle_model
        model = self.model

        oracle_init = oracle_model.predict_init_value(concat_input_list)
        oracle_init_mean, oracle_init_std = oracle_init[:, :self.input_size], oracle_init[:, self.input_size:]
        oracle_init_mean, oracle_init_std = unsqueeze(oracle_init_mean, dim=0), unsqueeze(oracle_init_std, dim=0)
        oracle_std = randn([sample_multiplier, batch_size, input_size]).to(device)
        oracle_init_value = oracle_init_mean + oracle_std * oracle_init_std

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

        oracle_init_value = oracle_init_value.reshape([self.batch_size * self.sample_multiplier, self.input_size])
        new_init_value = new_init_value.reshape([self.batch_size * self.sample_multiplier, self.input_size+1])

        oracle_predict_value = odeint(self.oracle_model.causal_derivative, oracle_init_value, time)
        new_predict_value = odeint(self.model.derivative, new_init_value, time)

        oracle_predict_value, new_predict_value = squeeze(oracle_predict_value), squeeze(new_predict_value)
        oracle_predict_value = oracle_predict_value.reshape([self.sample_multiplier, self.batch_size, self.input_size])
        new_predict_value = new_predict_value.reshape([self.sample_multiplier, self.batch_size, self.input_size+1])
        oracle_predict_value = permute(oracle_predict_value, (1, 0, 2))
        new_predict_value = permute(new_predict_value, (1, 0, 2))
        # 最后一个数据本质是不可观测的
        new_predict_value = new_predict_value[:, :, :input_size]

        loss_sum = 0
        oracle_predict_value_list = chunk(oracle_predict_value, chunks=self.batch_size, dim=0)
        new_predict_value_list = chunk(new_predict_value, chunks=self.batch_size, dim=0)
        for oracle, new in zip(oracle_predict_value_list, new_predict_value_list):
            loss = self.samples_loss(new, oracle)
            loss_sum += loss
        return loss_sum

    def set_treatment(self, idx, value, time):
        return self.model.derivative.set_treatment(idx, value, time)


class TrajectoryPrediction(Module):
    """
    这个其实是Causal Trajectory Prediction，但是在这里不需要特别复杂的因果分析设定了，所以结构可以简单一些
    """
    def __init__(self, oracle_adjacency, mode, treat_idx, input_size, hidden_size, bidirectional, batch_first,
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
        self.oracle_adjacency = oracle_adjacency
        self.derivative = Derivative(input_size, hidden_size, oracle_adjacency, device).to(device)


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


class Derivative(Module):
    def __init__(self, input_size, hidden_size, adjacency, device):
        super().__init__()
        self.input_size = input_size
        self.adjacency = adjacency
        self.device = device
        self.treatment_idx = None
        self.treatment_time = None
        self.treatment_value = None
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

    def set_treatment(self, treatment_idx, treatment_value, treatment_time):
        self.treatment_time = treatment_time
        self.treatment_value = treatment_value
        self.treatment_idx = treatment_idx

    def forward(self, t, inputs):
        # inputs shape [batch size, input dim]
        # 注意，再次求derivative一定会有一个隐变量，因此假设有input size+1

        if self.treatment_time is not None and t > self.treatment_time:
            inputs[:, self.treatment_idx] = self.treatment_value

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

        if self.treatment_time is not None and t > self.treatment_time:
            output_feature[:, self.treatment_idx] = 0
        return output_feature

