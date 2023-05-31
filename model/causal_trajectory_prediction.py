if __name__ == '__main__':
    print('unit test in verification')
import pickle
import os
from default_config import logger
from torch import FloatTensor, chunk, stack, squeeze, cat, transpose, eye, ones, no_grad, matmul, abs, sum, \
    trace, tanh, unsqueeze, LongTensor, randn, permute
from torch.linalg import matrix_exp
from torch.utils.data import RandomSampler
from torch.nn import Module, LSTM, Sequential, ReLU, Linear, MSELoss, ParameterList
from data_preprocess.data_loader import SequentialVisitDataloader, SequentialVisitDataset
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint
from datetime import datetime
import csv


class CausalTrajectoryPrediction(Module):
    def __init__(self, graph_type: str, constraint: str, input_size: int, hidden_size: int, batch_first: bool,
                 mediate_size: int, time_offset: int, clamp_edge_threshold: float, device: str, bidirectional: str,
                 dataset_name:str, mode:str):
        super().__init__()
        assert graph_type == 'DAG' or graph_type == 'ADMG'
        assert (graph_type == 'DAG' and constraint == 'default') or (constraint in {'ancestral', 'bow-free', 'arid'})
        assert bidirectional == 'True' or bidirectional == 'False'
        # mode指代输入的数据的时间间隔是固定的还是随机的，如果是固定的可以用序列化处理，随机的必须一条一条算
        assert mode == 'uniform' or 'random'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.graph_type = graph_type
        self.constraint = constraint
        self.time_offset = time_offset
        self.dataset_name = dataset_name
        self.clamp_edge_threshold = clamp_edge_threshold
        self.device = device
        self.init_net_bidirectional = True if bidirectional == 'True' else False
        self.mode = mode

        self.mse_loss_func = MSELoss(reduction='none')
        # init value estimate module
        self.init_network = LSTM(input_size=input_size*2+1, hidden_size=hidden_size, batch_first=batch_first,
                                 bidirectional=self.init_net_bidirectional)
        # 生成的分别是init value的均值与方差
        self.projection_net = Sequential(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, input_size * 2))
        self.causal_derivative = CausalDerivative(graph_type, constraint, input_size, hidden_size, mediate_size,
                                                  dataset_name, device, clamp_edge_threshold)

        self.init_network.to(device)
        self.projection_net.to(device)
        self.causal_derivative.to(device)

    def forward(self, data):
        # data format [batch_size, visit_idx, feature_idx]
        concat_input_list, _, _, _, label_feature_list, label_time_list, label_mask_list, label_type_list, _, _ = data

        # estimate the init value
        init = self.predict_init_value(concat_input_list)
        init_value, init_std = init[:, :self.input_size], init[:, self.input_size: ]
        std = randn(init_value.shape).to(self.device)
        init_value = init_value + std * init_std

        # predict label
        loss_sum, reconstruct_loss_sum, predict_loss_sum = 0.0, 0.0, 0.0
        if self.mode == 'random':
            predict_value_list = []
            init_value_list = chunk(init_value, init_value.shape[0], dim=0)
            for init_value, time, label, label_type, label_mask in \
                    zip(init_value_list, label_time_list, label_feature_list, label_type_list, label_mask_list):
                time = time - self.time_offset

                predict_value = squeeze(odeint(self.causal_derivative, init_value, time))
                predict_value_list.append(predict_value.detach())

                # 注意，label mask和label type指的是不一样的，前者指代的是每个element是否丢失，后者指代每个element是否有效
                sample_loss = self.mse_loss_func(predict_value, label) * (1 - label_mask)

                reconstruct_loss = sample_loss * unsqueeze(label_type == 1, dim=1)
                reconstruct_valid_ele_num = (reconstruct_loss != 0).sum()
                reconstruct_loss = reconstruct_loss.sum() / reconstruct_valid_ele_num
                reconstruct_loss_sum += reconstruct_loss

                predict_loss = sample_loss * unsqueeze(label_type == 2, dim=1)
                predict_valid_ele_num = (predict_loss != 0).sum()
                predict_loss = predict_loss.sum() / predict_valid_ele_num
                predict_loss_sum += predict_loss

            predict_loss_sum = predict_loss_sum / len(init_value_list)
            reconstruct_loss_sum = reconstruct_loss_sum / len(init_value_list)
            loss_sum = (reconstruct_loss_sum + predict_loss_sum) / 2
        else:
            assert self.mode == 'uniform'
            time = label_time_list[0] - self.time_offset
            predict_value = odeint(self.causal_derivative, init_value, time)
            predict_value = permute(predict_value, [1, 0, 2])
            label = stack(label_feature_list, dim=0)
            mask = stack(label_mask_list, dim=0)
            label_type = stack(label_type_list, dim=0)
            sample_loss = self.mse_loss_func(predict_value, label) * (1 - mask)

            reconstruct_loss = sample_loss * unsqueeze(label_type == 1, dim=2)
            reconstruct_valid_ele_num = (reconstruct_loss != 0).sum()
            # 此处计算ele时已经是element wise，无需像上面一样再除batch size
            reconstruct_loss_sum = reconstruct_loss.sum() / reconstruct_valid_ele_num

            predict_loss = sample_loss * unsqueeze(label_type == 2, dim=2)
            predict_valid_ele_num = (predict_loss != 0).sum()
            predict_loss_sum = predict_loss.sum() / predict_valid_ele_num

            predict_value_list = predict_value.detach()
            predict_value_list = chunk(predict_value_list, predict_value_list.shape[0], dim=0)

            loss_sum = (reconstruct_loss_sum + predict_loss_sum) / 2

        output_dict = {
            'predict_value_list': predict_value_list,
            'label_type_list': label_type_list,
            'label_feature_list': label_feature_list,
            'loss': loss_sum,
            'reconstruct_loss': reconstruct_loss_sum,
            'predict_loss': predict_loss_sum.detach()
        }
        return output_dict

    def predict_init_value(self, concat_input):
        length = LongTensor([len(item) for item in concat_input]).to(self.device)
        data = pad_sequence(concat_input, batch_first=True, padding_value=0).float()
        init_seq, _ = self.init_network(data)

        # 根据网上的相关资料，forward pass的index是前一半；这个slice的策略尽管我没查到，但是在numpy上试了试，似乎是对的
        if self.init_net_bidirectional:
            forward = init_seq[range(len(length)), length-1, :self.hidden_size]
            backward = init_seq[:, 0, self.hidden_size:]
            hidden_init = (forward + backward) / 2
        else:
            hidden_init = init_seq[range(len(length)), length-1, :]
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

    def calculate_constraint(self):
        return self.causal_derivative.graph_constraint()

    def clamp_edge(self):
        self.causal_derivative.clamp_edge(self.clamp_edge_threshold)

    def dump_graph(self, idx, folder=None):
        return self.causal_derivative.print_adjacency(idx, folder)

    def generate_binary_graph(self, threshold):
        return self.causal_derivative.generate_binary_adjacency(threshold)



class CausalDerivative(Module):
    def __init__(self, graph_type: str, constraint_type: str, input_size: int, hidden_size: int, mediate_size: int,
                 dataset_name: str, device: str, clamp_edge_threshold:float):
        super().__init__()
        assert graph_type == 'DAG' or graph_type == 'ADMG'
        self.constraint_type = constraint_type
        self.graph_type = graph_type
        self.input_size = input_size
        self.device = device
        self.directed_net_list = ParameterList()
        self.fuse_net_list = ParameterList()
        self.clamp_edge_threshold = clamp_edge_threshold
        self.dataset_name = dataset_name
        self.bi_directed_net_list = None
        if graph_type == 'DAG':
            for i in range(input_size):
                net_1 = Sequential(Linear(input_size, hidden_size, bias=False), ReLU(),
                                   Linear(hidden_size, mediate_size, bias=False), ReLU())
                self.directed_net_list.append(net_1)
                net_3 = Sequential(Linear(mediate_size + input_size, hidden_size), ReLU(),
                                   Linear(hidden_size, 1), ReLU())
                self.fuse_net_list.append(net_3)

            self.adjacency = {'dag': (ones([input_size, input_size]) - eye(input_size)).to(device)}
        elif graph_type == 'ADMG':
            self.bi_directed_net_list = ParameterList()
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
            self.adjacency = {
                'dag': (ones([input_size, input_size]) - eye(input_size)).to(device),
                'bi': (ones([input_size, input_size]) - eye(input_size)).to(device)
            }
        else:
            raise ValueError('')

    def generate_binary_adjacency(self, threshold):
        with no_grad():
            if self.graph_type == 'DAG':
                return {
                    'dag': (self.adjacency['dag'] > threshold).float().detach().to('cpu').numpy(),
                    'bi': None
                }
            elif self.graph_type == 'ADMG':
                return {
                    'dag': (self.adjacency['dag'] > threshold).float().detach().to('cpu').numpy(),
                    'bi': (self.adjacency['bi'] > threshold).float().detach().to('cpu').numpy()
                }
            else:
                raise ValueError('')

    def print_adjacency(self, iter_idx, write_folder=None):
        clamp_edge_threshold = self.clamp_edge_threshold
        with no_grad():
            if self.graph_type == 'DAG':
                dag_net_list = self.directed_net_list
                connect_mat = self.calculate_connectivity_mat(dag_net_list, absolute=True)
                self.adjacency['dag'] = connect_mat
                constraint = trace(matrix_exp((connect_mat > self.clamp_edge_threshold).float())) - self.input_size
            elif self.graph_type == 'ADMG':
                dag_net_list = self.directed_net_list
                dag_connect_mat = self.calculate_connectivity_mat(dag_net_list, absolute=True)
                self.adjacency['dag'] = dag_connect_mat
                bi_net_list = self.bi_directed_net_list
                bi_connect_mat = self.calculate_connectivity_mat(bi_net_list, absolute=True)
                self.adjacency['bi'] = bi_connect_mat
                assert self.constraint_type == 'ancestral'
                binary_connect_mat = (dag_connect_mat > self.clamp_edge_threshold).float()
                constraint = trace(matrix_exp(binary_connect_mat)) \
                             - self.input_size
                constraint = sum(((dag_connect_mat > self.clamp_edge_threshold) *
                                  (bi_connect_mat > self.clamp_edge_threshold)).float()) + constraint
            else:
                raise ValueError('')
        logger.info('binary graph constraint: {}'.format(constraint))

        write_content = [[self.graph_type]]
        if self.graph_type == 'DAG':
            adjacency = self.adjacency['dag'].to('cpu').numpy()
            logger.info('adjacency float')
            for item in adjacency:
                logger.info(item)
            logger.info('adjacency bool')
            for item in adjacency:
                logger.info(item > clamp_edge_threshold)
            for line in adjacency:
                line_content = []
                for item in line:
                    line_content.append(item)
                write_content.append(line_content)
        else:
            di = self.adjacency['dag'].to('cpu').numpy()
            bi = self.adjacency['bi'].to('cpu').numpy()
            logger.info('directed adjacency float')
            for item in di:
                logger.info(item)
            logger.info('directed adjacency bool')
            for item in di:
                logger.info(item > clamp_edge_threshold)
            logger.info('bi adjacency float')
            for item in bi:
                logger.info(item)
            logger.info('bi adjacency bool')
            for item in bi:
                logger.info(item > clamp_edge_threshold)
            write_content.append(['directed acyclic graph'])
            for line in di:
                line_content = []
                for item in line:
                    line_content.append(item)
                write_content.append(line_content)
            write_content.append(['bi-directed graph'])
            for line in bi:
                line_content = []
                for item in line:
                    line_content.append(item)
                write_content.append(line_content)

        if write_folder is not None:
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = '{}.{}.{}.{}.{}.csv'\
                .format(self.dataset_name, self.graph_type, self.constraint_type, iter_idx, now)
            write_path = os.path.join(write_folder, file_name)
            with open(write_path, 'w', encoding='utf-8-sig', newline='') as f:
                csv.writer(f).writerows(write_content)

    def clamp_edge(self, clamp_edge_threshold):
        with no_grad():
            if self.graph_type == 'DAG':
                dag_net_list = self.directed_net_list
                connect_mat = self.calculate_connectivity_mat(dag_net_list, absolute=True)
                keep_edge = connect_mat > clamp_edge_threshold
                self.adjacency['dag'] *= keep_edge
            elif self.graph_type == 'ADMG':
                dag_net_list = self.directed_net_list
                dag_connect_mat = self.calculate_connectivity_mat(dag_net_list, absolute=True)
                dag_keep_edge = dag_connect_mat > clamp_edge_threshold
                self.adjacency['dag'] *= dag_keep_edge
                bi_net_list = self.bi_directed_net_list
                bi_connect_mat = self.calculate_connectivity_mat(bi_net_list, absolute=True)
                bi_keep_edge = bi_connect_mat > clamp_edge_threshold
                self.adjacency['bi'] *= bi_keep_edge
            else:
                raise ValueError('')

    def forward(self, _, inputs):
        # designed for this format
        # inputs shape [batch size, input dim]
        assert inputs.shape[1] == self.input_size and len(inputs.shape) == 2
        inputs = unsqueeze(inputs, dim=1)
        inputs = inputs.repeat(1, inputs.shape[2], 1)
        assert inputs.shape[1] == inputs.shape[2] and len(inputs.shape) == 3

        # input_1 避免自环，input_2考虑自环
        filter_1 = unsqueeze(ones([inputs.shape[2], inputs.shape[2]]) - eye(inputs.shape[2]), dim=0).to(self.device)
        filter_2 = unsqueeze(eye(inputs.shape[2]), dim=0).to(self.device)
        inputs_1 = inputs * filter_1
        inputs_2 = inputs * filter_2

        output_feature = []
        if self.graph_type == 'DAG':
            adjacency = unsqueeze(self.adjacency['dag'], dim=0).to(self.device)
            inputs_1 = inputs_1 * adjacency
            inputs_1_list = chunk(inputs_1, inputs.shape[1], dim=1)
            inputs_2_list = chunk(inputs_2, inputs.shape[1], dim=1)
            for i in range(self.input_size):
                net_1 = self.directed_net_list[i]
                net_3 = self.fuse_net_list[i]
                input_1 = inputs_1_list[i]
                input_2 = inputs_2_list[i]

                representation_1 = net_1(input_1)
                representation_3 = cat([representation_1, input_2], dim=2)
                derivative = net_3(representation_3)
                output_feature.append(derivative)
        elif self.graph_type == 'ADMG':
            dag = unsqueeze(self.adjacency['dag'], dim=0).to(self.device)
            bi = unsqueeze(self.adjacency['bi'], dim=0).to(self.device)
            inputs_1_dag = inputs_1 * dag
            inputs_1_bi = inputs_1 * bi
            inputs_1_dag_list = chunk(inputs_1_dag, inputs.shape[1], dim=1)
            inputs_1_bi_list = chunk(inputs_1_bi, inputs.shape[1], dim=1)
            inputs_2_list = chunk(inputs_2, inputs.shape[1], dim=1)
            for i in range(self.input_size):
                net_1 = self.directed_net_list[i]
                net_2 = self.bi_directed_net_list[i]
                net_3 = self.fuse_net_list[i]
                input_1_dag = inputs_1_dag_list[i]
                input_1_bi = inputs_1_bi_list[i]
                input_2 = inputs_2_list[i]

                representation_1 = net_1(input_1_dag)
                representation_2 = net_2(input_1_bi)
                representation_3 = cat([representation_1, representation_2, input_2], dim=2)
                derivative = net_3(representation_3)
                output_feature.append(derivative)
        else:
            raise ValueError('')
        output_feature = cat(output_feature, dim=2)
        output_feature = squeeze(output_feature, dim=1)
        return output_feature

    def graph_constraint(self):
        # from arxiv 2010.06978, Table 1
        if self.graph_type == 'DAG':
            dag_net_list = self.directed_net_list
            directed_connect_mat = self.calculate_connectivity_mat(dag_net_list, absolute=True)
            constraint = trace(matrix_exp(directed_connect_mat)) - self.input_size
        elif self.graph_type == 'ADMG':
            dag_net_list = self.directed_net_list
            bi_net_list = self.bi_directed_net_list
            directed_connect_mat = self.calculate_connectivity_mat(dag_net_list, absolute=True)
            bi_connect_mat = self.calculate_connectivity_mat(bi_net_list, absolute=True)

            dag_constraint = trace(matrix_exp(directed_connect_mat)) - self.input_size
            if self.constraint_type == 'ancestral':
                bi_constraint = sum(matrix_exp(directed_connect_mat) * bi_connect_mat)
            elif self.constraint_type == 'bow-free':
                bi_constraint = sum(directed_connect_mat * bi_connect_mat)
            elif self.constraint_type == 'arid':
                bi_constraint = self.greenery(directed_connect_mat, bi_connect_mat)
            else:
                raise ValueError('')
            constraint = dag_constraint + bi_constraint
        else:
            raise ValueError('')
        assert constraint >= 0
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

    def calculate_connectivity_mat(self, module_list, absolute):
        # Note, C_ij means the i predicts the j
        # 此处的正确性已经经过校验
        # 注意Linear是算的xA^T，而此处是A直接参与计算，相关的形式和arXiv 1906.02226应该是完全一致的
        # 最后应该药转置
        connect_mat = []
        for i in range(len(module_list)):
            prod = eye(len(module_list)).to(self.device)
            parameters_list = [item for item in module_list[i].parameters()]
            for j in range(len(parameters_list)):
                if absolute:
                    para_mat = abs(parameters_list[j])
                else:
                    para_mat = parameters_list[j] * parameters_list[j]
                prod = matmul(para_mat, prod)
            connect_mat.append(prod)
        connect_mat = stack(connect_mat, dim=0)
        connect_mat = sum(connect_mat, dim=1)
        connect_mat = permute(connect_mat, [1, 0])
        return connect_mat


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
    clamp_edge_threshold = argument["clamp_edge_threshold"]
    bidirectional = argument['init_net_bidirectional']
    device = argument['device']
    dataset_name = argument['dataset_name']
    mode = argument['distribution_mode']

    model = CausalTrajectoryPrediction(graph_type=graph_type, constraint=constraint, input_size=input_size,
                                       hidden_size=hidden_size, batch_first=batch_first, mediate_size=mediate_size,
                                       time_offset=time_offset, clamp_edge_threshold=clamp_edge_threshold,
                                       bidirectional=bidirectional, device=device, dataset_name=dataset_name,
                                       mode=mode)
    _ = model.causal_derivative.graph_constraint()
    data = pickle.load(open(data_path, 'rb'))['data']['train']
    dataset = SequentialVisitDataset(data)
    sampler = RandomSampler(dataset)
    dataloader = SequentialVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                           minimum_observation=minimum_observation,
                                           reconstruct_input=reconstruct_input, predict_label=predict_label,
                                           device=device)
    idx = 0
    for batch in dataloader:
        logger.info('index: {}'.format(idx))
        idx += 1
        __ = model(batch)
