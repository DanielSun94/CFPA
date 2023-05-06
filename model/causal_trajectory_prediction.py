if __name__ == '__main__':
    print('unit test in verification')
import pickle
from default_config import logger
from torch import FloatTensor, chunk, stack, squeeze, cat, transpose, eye, ones, no_grad, matmul, abs, sum, \
    trace, tanh, unsqueeze, LongTensor
from torch.linalg import matrix_exp
from torch.nn.init import xavier_uniform_
from torch.utils.data import RandomSampler
from torch.nn import Module, LSTM, Sequential, ReLU, Linear, MSELoss
from data_preprocess.data_loader import SequentialVisitDataloader, SequentialVisitDataset
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint


class CausalTrajectoryPrediction(Module):
    def __init__(self, graph_type: str, constraint: str, input_size: int, hidden_size: int, batch_first: bool,
                 mediate_size: int, time_offset: int):
        super().__init__()
        assert graph_type == 'DAG' or graph_type == 'ADMG'
        assert (graph_type == 'DAG' and constraint == 'default') or (constraint in {'ancestral', 'bow-free', 'arid'})
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.__graph_type = graph_type
        self.__constraint = constraint
        self.time_offset = time_offset

        self.mse_loss_func = MSELoss(reduction='none')
        # init value estimate module
        self.init_network = LSTM(input_size=input_size*2+1, hidden_size=hidden_size, batch_first=batch_first,
                                 bidirectional=True)
        self.projection_net = Sequential(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, input_size))

        # casual derivative
        self.causal_derivative = CausalDerivative(graph_type, constraint, input_size, hidden_size, mediate_size)

    def forward(self, data):
        # data format [batch_size, visit_idx, feature_idx]
        concat_input_list, _, _, _, label_feature_list, label_time_list, label_mask_list, label_type_list, _, _ = data

        # estimate the init value
        init_value = self.predict_init_value(concat_input_list)

        # predict label
        init_value_list = chunk(init_value, init_value.shape[0], dim=0)
        predict_value_list = []
        loss_sum, reconstruct_loss_sum, predict_loss_sum = FloatTensor(0), FloatTensor(0), FloatTensor(0)
        for init_value, time, label, label_type, label_mask in\
                zip(init_value_list, label_time_list, label_feature_list, label_type_list, label_mask_list):
            time -= self.time_offset
            predict_value = squeeze(odeint(self.causal_derivative, init_value, time))
            predict_value_list.append(predict_value.detach())

            # 注意，label mask和label type指的是不一样的，前者指代的是每个element是否丢失，后者指代每个element是否有效
            sample_loss = self.mse_loss_func(predict_value, label) * (1-label_mask)

            reconstruct_loss = sample_loss * unsqueeze(label_type == 1, dim=1)
            reconstruct_valid_ele_num = (reconstruct_loss != 0).sum()
            reconstruct_loss = reconstruct_loss.sum() / reconstruct_valid_ele_num
            reconstruct_loss_sum += reconstruct_loss

            predict_loss = sample_loss * unsqueeze(label_type == 2, dim=1)
            predict_valid_ele_num = (predict_loss != 0).sum()
            predict_loss = predict_loss.sum() / predict_valid_ele_num
            predict_loss_sum += predict_loss

        predict_loss_sum = predict_loss_sum / len(init_value)
        reconstruct_loss_sum = reconstruct_loss_sum / len(init_value)
        loss_sum = (reconstruct_loss_sum + predict_loss_sum) / 2
        return predict_value_list, label_type_list, label_feature_list, loss_sum, \
            reconstruct_loss_sum.detach(), predict_loss_sum.detach()

    def predict_init_value(self, concat_input):
        length = LongTensor([len(item) for item in concat_input])
        concat_input = [FloatTensor(item) for item in concat_input]
        data = pad_sequence(concat_input, batch_first=True, padding_value=0).float()
        init_seq, _ = self.init_network(data)

        # 根据网上的相关资料，forward pass的index是前一半；这个slice的策略尽管我没查到，但是在numpy上试了试，似乎是对的
        forward = init_seq[range(len(length)), length-1, :self.hidden_size]
        backward = init_seq[:, 0, self.hidden_size:]
        hidden_init = (forward + backward) / 2
        value_init = self.projection_net(hidden_init)
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

    def constraint(self):
        return self.causal_derivative.graph_constraint()

    def generate_graph(self):
        # To Be Done
        raise NotImplementedError

    @staticmethod
    def init_weights(m):
        if isinstance(m, Linear):
            xavier_uniform_(m.weight)


class CausalDerivative(Module):
    def __init__(self, graph_type: str, constraint_type: str, input_size: int, hidden_size: int, mediate_size: int):
        super().__init__()
        assert graph_type == 'DAG' or graph_type == 'ADMG'
        self.constraint_type = constraint_type
        self.graph_type = graph_type
        self.input_size = input_size

        self.directed_net_list = []
        self.fuse_net_list = []
        self.bi_directed_net_list = None
        if graph_type == 'DAG':
            for i in range(input_size):
                net_1 = Sequential(Linear(input_size, hidden_size, bias=False), ReLU(),
                                   Linear(hidden_size, mediate_size, bias=False), ReLU())
                self.directed_net_list.append(net_1)
                net_3 = Sequential(Linear(mediate_size + input_size, hidden_size), ReLU(),
                                   Linear(hidden_size, 1), ReLU())
                self.fuse_net_list.append(net_3)
        elif graph_type == 'ADMG':
            self.bi_directed_net_list = []
            for i in range(input_size):
                net_1 = Sequential(Linear(input_size, hidden_size, bias=False), ReLU(),
                                   Linear(hidden_size, mediate_size, bias=False), ReLU())
                net_2 = Sequential(Linear(input_size, hidden_size, bias=False), ReLU(),
                                   Linear(hidden_size, mediate_size, bias=False), ReLU())
                self.directed_net_list.append(net_1)
                self.bi_directed_net_list.append(net_2)
                net_3 = Sequential(Linear(2 * mediate_size + input_size, hidden_size),
                                   ReLU(), Linear(hidden_size, 1), ReLU())
                self.fuse_net_list.append(net_3)
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
                net_3 = self.fuse_net_list[i]
                input_1 = inputs_1_list[i]
                input_2 = inputs_2_list[i]

                representation_1 = net_1(input_1)
                representation_3 = cat([representation_1, input_2], dim=1)
                derivative = net_3(representation_3)
                output_feature.append(derivative)
        elif self.graph_type == 'ADMG':
            for i in range(self.input_size):
                net_1 = self.directed_net_list[i]
                net_2 = self.bi_directed_net_list[i]
                net_3 = self.fuse_net_list[i]
                input_1 = inputs_1_list[i]
                input_2 = inputs_2_list[i]

                representation_1 = net_1(input_1)
                representation_2 = net_2(input_1)
                representation_3 = cat([representation_1, representation_2, input_2], dim=1)
                derivative = net_3(representation_3)
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


def unit_test(argument):
    data_path = argument['data_path']
    batch_first = True if argument['batch_first'] == 'True' else False
    batch_size = argument['batch_size']
    mediate_size = argument['mediate_size']
    minimum_observation = argument['minimum_observation']
    input_size = argument['input_size']
    mask_tag = argument['mask_tag']
    hidden_size = argument['hidden_size']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    graph_type = argument['graph_type']
    constraint = argument['constraint_type']
    time_offset = argument['time_offset']

    model = CausalTrajectoryPrediction(graph_type=graph_type, constraint=constraint, input_size=input_size,
                                       hidden_size=hidden_size, batch_first=batch_first, mediate_size=mediate_size,
                                       time_offset=time_offset)
    _ = model.causal_derivative.graph_constraint()
    data = pickle.load(open(data_path, 'rb'))['data']['train']
    dataset = SequentialVisitDataset(data)
    sampler = RandomSampler(dataset)
    dataloader = SequentialVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                           minimum_observation=minimum_observation,
                                           reconstruct_input=reconstruct_input, predict_label=predict_label)
    idx = 0
    for batch in dataloader:
        logger.info('index: {}'.format(idx))
        idx += 1
        __ = model(batch)
