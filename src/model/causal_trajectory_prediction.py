import os
import pickle
from torch import FloatTensor, chunk, stack, squeeze, cat, transpose, eye, ones, normal, no_grad, matmul, abs, sum, \
    trace, tanh, unsqueeze
from torch.linalg import matrix_exp
from torch.nn.init import xavier_uniform_
from torch.utils.data import RandomSampler
from torch.nn import Module, LSTM, Sequential, ReLU, Linear, MSELoss, BCELoss
from data_loader import SequentialVisitDataloader, SequentialVisitDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchdiffeq import odeint_adjoint as odeint


class CausalTrajectoryPrediction(Module):
    def __init__(self, graph_type: str, constraint: str, input_size: int, hidden_size: int, batch_first: bool,
                 mediate_size: int):
        super().__init__()
        assert graph_type == 'DAG' or graph_type == 'ADMG'
        assert (graph_type == 'DAG' and constraint == 'default') or (constraint in {'ancestral', 'bow-free', 'arid'})
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.__graph_type = graph_type
        self.__constraint = constraint

        # init value estimate module
        self.init_network = LSTM(input_size=input_size*2+1, hidden_size=hidden_size, batch_first=batch_first,
                                 bidirectional=True)
        self.projection_net = Sequential(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, input_size))

        # casual derivative
        self.causal_derivative = CausalDerivative(graph_type, constraint, input_size, hidden_size, mediate_size)

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
            predict_value.append(odeint(self.causal_derivative, init_value, time))
        predict_value = squeeze(stack(predict_value))

        # calculate loss
        loss, mse_loss, bce_loss = [], MSELoss(reduction='none'), BCELoss(reduction='none')
        predict_value = chunk(predict_value, predict_value.shape[1], dim=1)
        label_feature = chunk(label_feature, len(label_feature[0]), dim=1)
        for pred, label in zip(predict_value, label_feature):
            loss.append(mse_loss(pred, label))
        loss = transpose(squeeze(stack(loss, dim=0)), 0, 1)
        loss = loss * (1-label_mask)
        loss = loss.mean()
        return predict_value, label_mask_list, label_feature_list, loss

    def predict_init_value(self, concat_input):
        length = FloatTensor([len(item) for item in concat_input])
        concat_input = [FloatTensor(item) for item in concat_input]
        data = pad_sequence(concat_input, batch_first=True, padding_value=0).float()
        packed_data = pack_padded_sequence(data, length, batch_first=True, enforce_sorted=False)
        init_seq, _ = self.init_network(packed_data)
        out_pad, out_len = pad_packed_sequence(init_seq, batch_first=True)

        # 根据网上的相关资料，forward pass的index是前一半；这个slice的策略尽管我没查到，但是在numpy上试了试，似乎是对的
        forward = out_pad[range(len(out_pad)), out_len-1, :self.hidden_size]
        backward = out_pad[:, 0, self.hidden_size:]
        hidden_init = (forward + backward) / 2
        hidden_init = chunk(hidden_init, hidden_init.shape[0], dim=0)
        value_init_list = []
        for item in hidden_init:
            value_init_sample = self.projection_net(item)
            value_init_list.append(value_init_sample)
        value_init = squeeze(stack(value_init_list, dim=0))
        return value_init

    def inference(self, concat_input, time):
        with no_grad():
            predict_value = []
            init_value = self.predict_init_value(concat_input)
            label_time_list = FloatTensor(time)
            init_value_list = chunk(init_value, init_value.shape[0], dim=0)
            time_list = chunk(label_time_list, label_time_list.shape[0], dim=0)
            for init_value, time in zip(init_value_list, time_list):
                predict_value.append(odeint(self.causal_derivative, init_value, time))
            predict_value = squeeze(stack(predict_value))
        return predict_value


class CausalDerivative(Module):
    def __init__(self, graph_type: str, constraint_type: str, input_size: int, hidden_size: int, mediate_size: int):
        super().__init__()
        assert graph_type == 'DAG' or graph_type == 'ADMG'
        self.constraint_type = constraint_type
        self.graph_type = graph_type
        self.input_size = input_size

        self.directed_net_list = []
        self.self_excite_net_list = []
        self.fuse_net_list = []
        self.bi_directed_net_list = None
        if graph_type == 'DAG':
            for i in range(input_size):
                net_1 = Sequential(Linear(input_size, hidden_size, bias=False), ReLU(),
                                   Linear(hidden_size, mediate_size, bias=False), ReLU())
                net_2 = Sequential(Linear(input_size, 2), ReLU(), Linear(2, 1), ReLU())
                self.directed_net_list.append(net_1)
                self.self_excite_net_list.append(net_2)
                net_4 = Sequential(Linear(mediate_size + 1, hidden_size), ReLU(), Linear(hidden_size, 2), ReLU())
                self.fuse_net_list.append(net_4)

                net_1.apply(self.init_weights)
                net_2.apply(self.init_weights)
                net_4.apply(self.init_weights)
        elif graph_type == 'ADMG':
            self.bi_directed_net_list = []
            for i in range(input_size):
                net_1 = Sequential(Linear(input_size, hidden_size, bias=False), ReLU(),
                                   Linear(hidden_size, mediate_size, bias=False), ReLU())
                net_2 = Sequential(Linear(input_size, 2), ReLU(), Linear(2, 1), ReLU())
                net_3 = Sequential(Linear(input_size, hidden_size, bias=False), ReLU(),
                                   Linear(hidden_size, mediate_size, bias=False), ReLU())
                self.directed_net_list.append(net_1)
                self.self_excite_net_list.append(net_2)
                self.bi_directed_net_list.append(net_3)
                net_4 = Sequential(Linear(2 * mediate_size + 1, hidden_size), ReLU(), Linear(hidden_size, 2), ReLU())
                self.fuse_net_list.append(net_4)

                net_1.apply(self.init_weights)
                net_2.apply(self.init_weights)
                net_3.apply(self.init_weights)
                net_4.apply(self.init_weights)
        else:
            raise ValueError('')

    def forward(self, _, inputs):
        # designed for this format
        assert inputs.shape[1] == self.input_size
        assert inputs.shape[0] == 1 and len(inputs.shape) == 2
        inputs = inputs.repeat(inputs.shape[1], 1)
        assert inputs.shape[0] == inputs.shape[1] and len(inputs.shape) == 2

        diag_mat = eye(inputs.shape[0])

        inputs_1 = inputs * (ones([inputs.shape[0], inputs.shape[0]]) - diag_mat)
        inputs_2 = inputs * diag_mat
        inputs_1_list = chunk(inputs_1, inputs.shape[0], dim=0)
        inputs_2_list = chunk(inputs_2, inputs.shape[0], dim=0)

        output_feature = []
        if self.graph_type == 'DAG':
            for i in range(self.input_size):
                net_1 = self.directed_net_list[i]
                net_2 = self.self_excite_net_list[i]
                net_4 = self.fuse_net_list[i]
                input_1 = inputs_1_list[i]
                input_2 = inputs_2_list[i]

                representation_1 = net_1(input_1)
                representation_2 = net_2(input_2)
                representation_3 = cat([representation_1, representation_2], dim=1)
                predict = net_4(representation_3)

                sample = normal(0, 1, [1])
                predict_mean, predict_std = chunk(predict, 2, dim=1)
                derivative = predict_mean + sample * predict_std
                output_feature.append(derivative)
        elif self.graph_type == 'ADMG':
            for i in range(self.input_size):
                net_1 = self.directed_net_list[i]
                net_2 = self.self_excite_net_list[i]
                net_3 = self.bi_directed_net_list[i]
                net_4 = self.fuse_net_list[i]
                input_1 = inputs_1_list[i]
                input_2 = inputs_2_list[i]

                representation_1 = net_1(input_1)
                representation_2 = net_3(input_1)
                representation_3 = net_2(input_2)
                representation_4 = cat([representation_1, representation_2, representation_3], dim=1)
                predict = net_4(representation_4)
                sample = normal(0, 1, [1])
                predict_mean, predict_std = chunk(predict, 2, dim=1)
                derivative = predict_mean + sample * predict_std
                output_feature.append(derivative)
        else:
            raise ValueError('')
        output_feature = cat(output_feature, dim=1)
        return output_feature

    def graph_constraint(self):
        # from arxiv 2010.06978, Table 1
        if self.graph_type == 'DAG':
            dag_net_list = self.directed_net_list
            directed_connect_mat = self.calculate_connectivity_mat(dag_net_list)
            constraint = trace(matrix_exp(directed_connect_mat)) - self.input_size
            assert constraint > 0
        elif self.graph_type == 'ADMG':
            dag_net_list = self.directed_net_list
            bi_net_list = self.bi_directed_net_list
            directed_connect_mat = self.calculate_connectivity_mat(dag_net_list)
            bi_connect_mat = self.calculate_connectivity_mat(bi_net_list)

            dag_constraint = trace(matrix_exp(directed_connect_mat)) - self.input_size
            if self.constraint_type == 'ancestral':
                bi_constraint = sum(matrix_exp(directed_connect_mat) * bi_connect_mat)
            elif self.constraint_type == 'bow-free':
                bi_constraint = sum(directed_connect_mat * bi_connect_mat)
            elif self.constraint_type == 'arid':
                bi_constraint = self.greenery(directed_connect_mat, bi_connect_mat)
            else:
                raise ValueError('')
            assert bi_constraint > 0
            constraint = dag_constraint + bi_constraint
        else:
            raise ValueError('')
        return constraint

    def greenery(self, directed_connect_mat, bi_connect_mat):
        # from arxiv 2010.06978, Algorithm 1
        greenery = -self.input_size
        identity = eye(self.input_size)
        for i in range(self.input_size):
            d_f, b_f = directed_connect_mat, bi_connect_mat
            for j in range(1, self.input_size-1):
                t = transpose(sum(matrix_exp(b_f) * d_f, dim=1, keepdim=True), 0, 1)
                identity_i = unsqueeze(identity[i, :], dim=0)
                f = tanh(t + identity_i)
                f_mat = transpose(f, 0, 1)
                f_mat = transpose(f_mat.repeat([1, self.input_size]), 0, 1)
                d_f = d_f * f_mat
                b_f = b_f * f_mat * transpose(f_mat, 0, 1)
                c_mat = matrix_exp(b_f) * matrix_exp(d_f)
                greenery = greenery + sum(c_mat[:, i])
        return greenery

    @staticmethod
    def calculate_connectivity_mat(module_list):
        # Note, C_ij means the i predicts the j
        # 此处注意符合正确性
        connect_mat = []
        for i in range(len(module_list)):
            prod = eye(len(module_list))
            parameters_list = [item for item in module_list[i].parameters()]
            for j in range(len(parameters_list)):
                para_mat = abs(parameters_list[j])
                prod = matmul(para_mat, prod)
            connect_mat.append(prod)
        connect_mat = sum(stack(connect_mat, dim=0), dim=1)
        connect_mat = transpose(connect_mat, 0, 1)
        return connect_mat


    @staticmethod
    def init_weights(m):
        if isinstance(m, Linear):
            xavier_uniform_(m.weight)


def unit_test():
    data_folder = os.path.abspath('../../resource/simulated_data/')
    hidden_false_data = os.path.join(data_folder, 'sim_data_hidden_False_group_lmci_personal_0_type_random.pkl')
    hidden_true_data = os.path.join(data_folder, 'sim_data_hidden_True_group_lmci_personal_0_type_random.pkl')

    batch_first = True
    batch_size = 128
    mediate_size = 2
    minimum_observation = 2
    input_size = 5
    mask_tag = -1
    hidden_size = 4
    reconstruct_input = True
    predict_label = True

    test_seq = [
        ['ADMG', 'arid'],
        ['DAG', 'default'],
        ['ADMG', 'ancestral'],
        ['ADMG', 'bow-free'],
    ]

    for (graph_type, constraint) in test_seq:
        model = CausalTrajectoryPrediction(graph_type=graph_type, constraint=constraint, input_size=input_size,
                                           hidden_size=hidden_size, batch_first=batch_first, mediate_size=mediate_size)
        loss = model.causal_derivative.graph_constraint()
        if graph_type == 'DAG':
            data = pickle.load(open(hidden_false_data, 'rb'))['data']['train']
        elif graph_type == 'ADMG':
            data = pickle.load(open(hidden_true_data, 'rb'))['data']['train']
        else:
            raise ValueError('')
        dataset = SequentialVisitDataset(data)
        sampler = RandomSampler(dataset)
        dataloader = SequentialVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                               minimum_observation=minimum_observation,
                                               reconstruct_input=reconstruct_input, predict_label=predict_label)
        for batch in dataloader:
            out = model(batch)


if __name__ == '__main__':
    unit_test()
