import numpy as np
import torch
from sklearn.metrics import roc_auc_score
if __name__ == '__main__':
    print('unit test in verification')
from default_config import logger
from torch import chunk, stack, squeeze, cat, eye, ones, no_grad, matmul, abs, sum, \
    trace, unsqueeze, LongTensor, randn, permute, FloatTensor, zeros
from torch.linalg import matrix_exp
from torch.nn import Module, LSTM, Sequential, ReLU, Linear, MSELoss, ParameterList, BCEWithLogitsLoss, Sigmoid, Softmax
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint


class TrajectoryPrediction(Module):
    def __init__(self, hidden_flag: str, constraint: str, input_size: int, hidden_size: int, batch_first: str,
                 time_offset: int, clamp_edge_threshold: float, device: str, bidirectional: str, non_linear_mode: str,
                 dataset_name: str, data_mode: str, process_name:str, input_type_list: list):
        super().__init__()
        assert hidden_flag == 'False' or hidden_flag == 'True'
        assert constraint in {'DAG', 'sparse', 'none'}
        assert bidirectional == 'True' or bidirectional == 'False'
        assert batch_first == 'True' or batch_first == 'False'
        assert non_linear_mode == 'True' or non_linear_mode == 'False'
        # mode指代输入的数据的时间间隔是固定的还是随机的，如果是固定的可以用序列化处理，随机的必须一条一条算
        assert data_mode == 'uniform' or 'random'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_flag = True if hidden_flag == 'True' else False
        self.constraint = constraint
        self.time_offset = time_offset
        self.sample_multiplier = 1
        self.input_type_list = input_type_list
        self.dataset_name = dataset_name
        self.clamp_edge_threshold = clamp_edge_threshold
        self.process_name = process_name
        self.device = device
        self.init_net_bidirectional = True if bidirectional == 'True' else False
        self.data_mode = data_mode
        self.non_linear_mode = True if non_linear_mode == 'True' else False
        self.batch_first = True if batch_first == 'True' else False
        self.mse_loss_func = MSELoss(reduction='none')
        self.cross_entropy_func = BCEWithLogitsLoss(reduction='none')

        # 生成的分别是init value的均值与方差
        if self.hidden_flag:
            self.derivative_dim = input_size + 1
        else:
            self.derivative_dim = input_size

        self.init_network = LSTM(input_size=self.input_size * 2 + 1, hidden_size=hidden_size,
                                 batch_first=self.batch_first, bidirectional=self.init_net_bidirectional)
        self.project_net = Linear(hidden_size, self.derivative_dim*2)

        if self.hidden_flag:
            input_type_list = input_type_list + ['continuous']
        self.derivative = CausalDerivative(constraint, self.derivative_dim, hidden_size, dataset_name, device,
                                           clamp_edge_threshold, input_type_list, self.hidden_flag,
                                           self.non_linear_mode)
        self.sigmoid = Sigmoid()
        self.project_net.to(device)
        self.init_network.to(device)
        self.derivative.to(device)
        self.consistent_feature = self.get_consistent_feature_flag()

    def set_adjacency_graph(self, adjacency):
        self.derivative.set_adjacency(adjacency)

    def get_consistent_feature_flag(self):
        input_type_list = self.input_type_list
        idx = 0
        for item in input_type_list:
            if item == 'continuous':
                idx += 0
            elif item == 'discrete':
                idx += 1
            else:
                raise ValueError('')
        if idx == len(input_type_list):
            return 'discrete'
        elif idx == 0:
            return 'continuous'
        else:
            return 'none'

    def set_sample_multiplier(self, num):
        self.sample_multiplier = num

    def forward(self, concat_input_list, label_time_list):
        # data format [batch_size, visit_idx, feature_idx]
        batch_size = len(concat_input_list)

        # estimate the init value
        init = self.predict_init_value(concat_input_list)
        init_value, init_std = init[:, :, 0], init[:, :, 1]
        std = randn([self.sample_multiplier, init_value.shape[0], init_value.shape[1]]).to(self.device)
        init_value, init_std = unsqueeze(init_value, dim=0), unsqueeze(init_std, dim=0)
        init_value = init_value + std * init_std
        init_value = init_value.reshape([batch_size * self.sample_multiplier, self.derivative_dim])

        if self.data_mode == 'random':
            predict_value_list = []
            init_value_list = chunk(init_value, init_value.shape[0], dim=0)
            label_time_list_multiplier = []
            for item in label_time_list:
                for i in range(self.sample_multiplier):
                    label_time_list_multiplier.append(item)
            for init_value, time in zip(init_value_list, label_time_list_multiplier):
                time = (time - self.time_offset).to(self.device)
                predict_value = squeeze(odeint(self.derivative, init_value, time))
                predict_value_list.append(predict_value)
        else:
            assert self.data_mode == 'uniform'
            time = (label_time_list[0] - self.time_offset).to(self.device)
            predict_value = odeint(self.derivative, init_value, time)
            predict_value = permute(predict_value, [1, 0, 2])
            predict_value = chunk(predict_value, predict_value.shape[0], dim=0)
            predict_value_list = []
            for i in range(len(predict_value)):
                predict_value_list.append(squeeze(predict_value[i], dim=0))
        return predict_value_list

    def performance_eval(self, prediction_list, feature_list, mask_list, type_list):
        input_type_list = self.input_type_list
        consistent = self.consistent_feature
        if consistent != 'none' and input_type_list[0] == 'continuous':
            output_dict = self.loss_calculate(prediction_list, feature_list, mask_list, type_list)
            output_dict['auc'] = 0
            output_dict['reconstruct_auc'] = 0
            output_dict['predict_auc'] = 0
            output_dict['mse'] = output_dict['loss'].detach().to('cpu').numpy()
            output_dict['reconstruct_mse'] = output_dict['reconstruct_loss'].detach().to('cpu').numpy()
            output_dict['predict_mse'] = output_dict['predict_loss'].detach().to('cpu').numpy()
            return output_dict

        full = {i: {'pred': [], 'label': [], 'type': input_type_list[i]} for i in range(len(input_type_list))}
        prediction = {i: {'pred': [], 'label': [], 'type': input_type_list[i]} for i in range(len(input_type_list))}
        reconstruct = {i: {'pred': [], 'label': [], 'type': input_type_list[i]} for i in range(len(input_type_list))}
        for mask, pred, label, visit_type_list in zip(mask_list, prediction_list, feature_list, type_list):
            for i, feature_type in enumerate(input_type_list):
                assert feature_type == 'continuous' or feature_type == 'discrete'
                for j in range(visit_type_list.shape[0]):
                    visit_type = float(visit_type_list[j].detach().to('cpu').numpy())
                    assert visit_type == 1 or visit_type == 2
                    feature_mask = float(mask[j, i].detach().to('cpu').numpy())
                    feature_label = float(label[j, i].detach().to('cpu').numpy())
                    feature_predict = float(pred[j, i].detach().to('cpu').numpy())
                    if feature_mask == 1:
                        continue
                    if visit_type == 1:
                        prediction[i]['pred'].append(feature_predict)
                        prediction[i]['label'].append(feature_label)
                    if visit_type == 2:
                        reconstruct[i]['pred'].append(feature_predict)
                        reconstruct[i]['label'].append(feature_label)
                    full[i]['pred'].append(feature_predict)
                    full[i]['label'].append(feature_label)

        auc, pred_auc, recons_auc, mse, pred_mse, recons_mse = [], [], [], [], [], []
        for i in range(len(input_type_list)):
            full_pred, full_label = FloatTensor(np.array(full[i]['pred'])), FloatTensor(np.array(full[i]['label']))
            reconstruct_pred, reconstruct_label = \
                FloatTensor(np.array(reconstruct[i]['pred'])), FloatTensor(np.array(reconstruct[i]['label']))
            predict_pred, predict_label = \
                FloatTensor(np.array(prediction[i]['pred'])), FloatTensor(np.array(prediction[i]['label']))
            if input_type_list[i] == 'continuous':
                mse.append(np.mean(self.mse_loss_func(full_pred, full_label).to('cpu').numpy()))
                pred_mse.append(np.mean(self.mse_loss_func(predict_pred, predict_label).to('cpu').numpy()))
                recons_mse.append(np.mean(self.mse_loss_func(reconstruct_pred, reconstruct_label).to('cpu').numpy()))
            else:
                if torch.sum(full_label) > 0:
                    full_pred = self.sigmoid(full_pred)
                    full_pred, full_label = full_pred.to('cpu').numpy(), full_label.to('cpu').numpy()
                    auc.append(roc_auc_score(full_label, full_pred))
                if torch.sum(reconstruct_label) > 0:
                    reconstruct_pred = self.sigmoid(reconstruct_pred)
                    reconstruct_pred, reconstruct_label = \
                        reconstruct_pred.to('cpu').numpy(), reconstruct_label.to('cpu').numpy()
                    recons_auc.append(roc_auc_score(reconstruct_label, reconstruct_pred))
                if torch.sum(predict_label) > 0:
                    predict_pred = self.sigmoid(predict_pred)
                    predict_pred, predict_label = predict_pred.to('cpu').numpy(), predict_label.to('cpu').numpy()
                    pred_auc.append(roc_auc_score(predict_label, predict_pred))

        auc = np.mean(auc)
        pred_auc = np.mean(pred_auc)
        recons_auc = np.mean(recons_auc)
        mse = np.mean(mse)
        pred_mse = np.mean(pred_mse)
        recons_mse = np.mean(recons_mse)
        output_dict = {'auc': auc, 'reconstruct_auc': recons_auc, 'predict_auc': pred_auc, 'mse': mse,
                       'reconstruct_mse': recons_mse, 'predict_mse': pred_mse}
        return output_dict


    def loss_calculate(self, prediction_list, feature_list, mask_list, type_list):
        # 按照设计，这个函数只有在预测阶段有效，预测阶段的multiplier必须为1
        assert self.sample_multiplier == 1
        consistent_feature = self.consistent_feature
        if self.data_mode == 'uniform' and consistent_feature != 'none':
            # 输出的predict_value_list是multiplier*batch size长度的序列
            prediction_list = stack(prediction_list, dim=0)
            if self.hidden_flag:
                prediction_list = prediction_list[:, :, :-1]
            assert len(prediction_list.shape) == 3 and prediction_list.shape[2] == self.input_size

            if consistent_feature == 'continuous':
                loss_func = self.mse_loss_func
            elif consistent_feature == 'discrete':
                loss_func = self.cross_entropy_func
            else:
                raise ValueError('')

            feature_list = stack(feature_list, dim=0)
            mask_list = stack(mask_list, dim=0)
            type_list = unsqueeze(stack(type_list, dim=0), dim=2)

            sample_loss = loss_func(prediction_list, feature_list) * (1 - mask_list)
            reconstruct_loss = sample_loss * (type_list == 1).float()
            predict_loss = sample_loss * (type_list == 2).float()
            predict_valid_ele_num = (predict_loss != 0).sum()
            reconstruct_valid_ele_num = (reconstruct_loss != 0).sum()
            reconstruct_loss = reconstruct_loss.sum() / reconstruct_valid_ele_num
            predict_loss = predict_loss.sum() / predict_valid_ele_num

            adapt_ratio = reconstruct_loss.detach() / predict_loss.detach()
            loss = adapt_ratio * reconstruct_loss + predict_loss
            output_dict = {
                'predict_value_list': prediction_list,
                'label_type_list': type_list,
                'label_feature_list': feature_list,
                'loss': loss,
                'reconstruct_loss': reconstruct_loss.detach(),
                'predict_loss': predict_loss.detach()
            }
        else:
            # 这里必须做双循环，因为random模式下每个sample是不定长的，在当前设计下不能tensor化
            loss_sum, predict_loss_sum, reconstruct_loss_sum = 0, 0, 0
            for predict, label, mask, type_ in zip(prediction_list, feature_list, mask_list, type_list):
                for i in range(len(self.input_type_list)):
                    data_type = self.input_type_list[i]
                    feature_pred, label_pred, mask_pred = predict[:, i], label[:, i], (1-mask[:, i])
                    if data_type == 'continuous':
                        sample_loss = self.mse_loss_func(feature_pred, label_pred) * mask_pred
                    elif data_type == 'discrete':
                        sample_loss = self.cross_entropy_func(feature_pred, label_pred) * mask_pred
                    else:
                        raise ValueError('')

                    reconstruct_loss = sample_loss * (type_ == 1).float()
                    reconstruct_valid_ele_num = (reconstruct_loss != 0).sum()
                    if reconstruct_valid_ele_num > 0:
                        reconstruct_loss = reconstruct_loss.sum() / reconstruct_valid_ele_num
                    else:
                        reconstruct_loss = 0

                    predict_loss = sample_loss * (type_ == 2).float()
                    predict_valid_ele_num = (predict_loss != 0).sum()
                    if predict_valid_ele_num > 0:
                        predict_loss = predict_loss.sum() / predict_valid_ele_num
                    else:
                        predict_loss = 0

                    loss = (reconstruct_loss + predict_loss) / 2

                    loss_sum = loss + loss_sum
                    predict_loss_sum = predict_loss_sum + predict_loss
                    reconstruct_loss_sum = reconstruct_loss + reconstruct_loss_sum

            loss_sum = loss_sum / len(prediction_list) / len(self.input_type_list)
            reconstruct_loss_sum = reconstruct_loss_sum / len(prediction_list) / len(self.input_type_list)
            predict_loss_sum = predict_loss_sum / len(prediction_list) / len(self.input_type_list)
            output_dict = {
                'predict_value_list': prediction_list,
                'label_type_list': type_list,
                'label_feature_list': feature_list,
                'loss': loss_sum,
                'reconstruct_loss': reconstruct_loss_sum.detach(),
                'predict_loss': predict_loss_sum.detach()
            }
        return output_dict

    def predict_init_value(self, concat_input):
        length = LongTensor([len(item) for item in concat_input]).to(self.device)
        data = pad_sequence(concat_input, batch_first=True, padding_value=0).float()
        init_seq, _ = self.init_network(data)

        if self.init_net_bidirectional:
            forward = init_seq[range(len(length)), length - 1, :self.hidden_size]
            backward = init_seq[:, 0, self.hidden_size:]
            hidden_init = (forward + backward) / 2
        else:
            hidden_init = init_seq[range(len(length)), length - 1, :]
        value_init = self.project_net(hidden_init)
        shape = value_init.shape
        value_init = value_init.reshape([shape[0], shape[1] // 2, 2])
        return value_init

    def calculate_constraint(self):
        return self.derivative.graph_constraint(), self.derivative.sparse_constraint()

    def set_adjacency(self, oracle):
        self.derivative.set_adjacency(oracle)

    def clamp_edge(self):
        self.derivative.clamp_edge(self.clamp_edge_threshold)

    def print_graph(self, idx, folder=None):
        return self.derivative.print_adjacency(idx, folder)

    def generate_binary_graph(self, threshold):
        return self.derivative.generate_binary_adjacency(threshold)


class Derivative(Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, inputs):
        raise NotImplementedError

    def print_adjacency(self, iter_idx, write_folder=None):
        raise NotImplementedError

    def clamp_edge(self, clamp_edge_threshold):
        raise NotImplementedError

    def graph_constraint(self):
        raise NotImplementedError

    def calculate_connectivity_mat(self, module_list, absolute):
        raise NotImplementedError


class CausalDerivative(Derivative):
    def __init__(self, constraint_type: str, input_size: int, hidden_size: int, dataset_name: str, device: str,
                 clamp_edge_threshold: float, input_type_list: list, hidden_flag:bool, non_linear_mode: bool):
        super().__init__()
        self.constraint_type = constraint_type
        self.input_size = input_size
        self.device = device
        self.hidden_flag = hidden_flag
        self.net_list = ParameterList()
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=1)
        self.clamp_edge_threshold = clamp_edge_threshold
        self.dataset_name = dataset_name
        self.input_type_list = unsqueeze(FloatTensor(
            np.array([1 if item == 'continuous' else 0 for item in input_type_list])), dim=0).to(device)
        self.non_linear_mode = non_linear_mode

        self.treatment_idx = None
        self.treatment_value = None
        self.treatment_time = None

        self.net_list = ParameterList()
        if non_linear_mode:
            for i in range(self.input_size):
                self.net_list.append(
                    Sequential(
                        Linear(self.input_size, hidden_size, bias=False),
                        # ReLU(),
                        Linear(hidden_size, hidden_size, bias=False),
                        # ReLU(),
                        Linear(hidden_size, 1, bias=False),
                    )
                )
        else:
            for i in range(self.input_size):
                self.net_list.append(
                    Sequential(
                        Linear(self.input_size, 1, bias=False),
                    )
                )

        self.net_list.to(device)
        self.adjacency = (ones([input_size, input_size])).to(device)

    def set_adjacency(self, adjacency):
        self.adjacency = FloatTensor(adjacency).to(self.device)

    def set_treatment(self, treatment_idx, treatment_value, treatment_time):
        self.treatment_time = treatment_time
        self.treatment_value = treatment_value
        self.treatment_idx = treatment_idx

    def clamp_edge(self, clamp_edge_threshold):
        with no_grad():
            dag_net_list = self.net_list
            connect_mat = self.calculate_connectivity_mat(dag_net_list, absolute=True)
            keep_edge = connect_mat > clamp_edge_threshold
            self.adjacency *= keep_edge

    def print_adjacency(self, iter_idx, write_folder=None):
        with no_grad():
            net_list = self.net_list
            if len(net_list) < 8:
                connect_mat = self.calculate_connectivity_mat(net_list, absolute=True)
                constraint = trace(matrix_exp((connect_mat > self.clamp_edge_threshold).float())) - self.input_size
                logger.info('binary graph constraint: {}'.format(constraint))
                connect_mat = connect_mat.to('cpu').numpy()
                logger.info('adjacency float')
                for item in connect_mat:
                    item_str_list = []
                    for key in item:
                        key = float(key)
                        item_str_list.append('{:>9.9f}'.format(key))
                    item_str = ', '.join(item_str_list)
                    logger.info('[' + item_str + ']')
            else:
                logger.info('adjacency print skipped for its size')

    def forward(self, t, inputs):
        # designed for this format
        # inputs shape [batch size, input dim]
        assert inputs.shape[1] == self.input_size and len(inputs.shape) == 2
        input_type_list = self.input_type_list

        assert (self.treatment_time is None and self.treatment_idx is None and self.treatment_value is None) or \
               (self.treatment_time is not None and self.treatment_idx is not None and self.treatment_value is not None)

        # y_hard = (inputs > 0).float()
        # y_soft = self.sigmoid(inputs)
        # inputs_discrete = y_hard - y_soft.detach() + y_soft
        inputs_discrete = self.sigmoid(inputs)
        inputs = inputs_discrete * (1-input_type_list) + inputs *  input_type_list

        # 当存在设定treatment时，强行赋值
        if self.treatment_time is not None and t > self.treatment_time:
            inputs[:, self.treatment_idx] = self.treatment_value

        inputs = unsqueeze(inputs, dim=1)
        inputs = inputs.repeat(1, inputs.shape[2], 1)
        assert inputs.shape[1] == inputs.shape[2] and len(inputs.shape) == 3

        output_feature = []
        # 因为设计原因，这里需要再转置一次，才是正确的导数计算方法
        transpose_adjacency = permute(self.adjacency, (1, 0))
        adjacency = unsqueeze(transpose_adjacency, dim=0).to(self.device)
        inputs = inputs * adjacency
        inputs_list = chunk(inputs, inputs.shape[1], dim=1)
        for i in range(self.input_size):
            net_1 = self.net_list[i]
            input_1 = inputs_list[i]

            # 这一情况下最后一个item默认是隐变量，不考虑其它变量预测
            if i == self.input_size-1 and self.hidden_flag:
                feature_filter = zeros([1, 1, self.input_size]).to(self.device)
                feature_filter[0, 0, -1] = 1
                input_1 = input_1 * feature_filter

            derivative = net_1(input_1)
            output_feature.append(derivative)

        output_feature = cat(output_feature, dim=2)
        output_feature = squeeze(output_feature, dim=1)

        if self.treatment_time is not None and t > self.treatment_time:
            output_feature[:, self.treatment_idx] = 0
        return output_feature

    def sparse_constraint(self):
        net_list = self.net_list
        connect_mat = self.calculate_connectivity_mat(net_list, absolute=True)
        connect_mat = (1 - eye(len(net_list))).to(self.device) * connect_mat
        if self.hidden_flag:
            loss_observed = sum(connect_mat[:-1,:])
            loss_hidden = 30 * sum(connect_mat[-1,:])
            loss = loss_hidden + loss_observed
        else:
            loss = sum(connect_mat)
        return loss

    def graph_constraint(self):
        # from arxiv 2010.06978, Table 1
        net_list = self.net_list
        connect_mat_origin = self.calculate_connectivity_mat(net_list, absolute=True)
        if self.hidden_flag:
            valid_size = self.input_size - 1
            connect_mat = connect_mat_origin[:-1, :-1]
        else:
            valid_size = self.input_size
            connect_mat = connect_mat_origin
        connect_mat = (1 - eye(valid_size)).to(self.device) * connect_mat
        constraint = trace(matrix_exp(connect_mat)) - valid_size
        return constraint

    def calculate_connectivity_mat(self, module_list, absolute):
        # Note, C_ij means the i predicts the j
        # 此处的正确性已经经过校验
        # 注意Linear是算的xA^T，而此处是A直接参与计算，相关的形式和arXiv 1906.02226应该是完全一致的
        # 最后应该转置
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

