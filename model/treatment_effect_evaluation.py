import torch
import os
from default_config import ckpt_folder
from model.causal_trajectory_prediction import TrajectoryPrediction
from torch import FloatTensor, chunk, stack, squeeze, cdist
from torch.nn import Module
from torch.nn import ModuleList
from torchdiffeq import odeint_adjoint as odeint


class TreatmentEffectEstimator(Module):
    def __init__(self, dataset_name, device, batch_size, new_model_number, oracle_graph, treatment_idx,
                 treatment_value, treatment_time, treatment_feature, id_type_list, model_args):
        super().__init__()

        # treatment time为-1，-2时表示施加干预的时间是最后一次入院的时间/倒数第二次入院的时间，是正数时代表任意指定时间
        self.model_args = model_args
        self.dataset_name = dataset_name
        self.oracle_graph = oracle_graph
        self.batch_size = batch_size
        self.new_model_number = new_model_number
        self.device = device
        self.treatment_feature = treatment_feature
        self.treatment_idx = treatment_idx
        self.treatment_value = treatment_value
        self.treatment_time = treatment_time
        self.id_type_list = id_type_list
        self.models = self.build_models(new_model_number, oracle_graph)\

    def build_models(self, new_model_number, oracle_graph):
        model_list = ModuleList()
        hidden_flag = self.model_args['hidden_flag']
        input_size = self.model_args['input_size']
        data_mode = self.model_args['distribution_mode']
        hidden_size = self.model_args['hidden_size']
        batch_first = self.model_args['batch_first']
        bidirectional = self.model_args['bidirectional']
        device = self.model_args['device']
        dataset_name = self.model_args['dataset_name']
        time_offset = self.model_args['time_offset']
        id_type_list = self.id_type_list

        for i in range(new_model_number):
            model = TrajectoryPrediction(
                hidden_flag=hidden_flag, constraint='none', input_size=input_size, hidden_size=hidden_size,
                mode=data_mode, batch_first=batch_first, time_offset=time_offset, input_type_list=id_type_list,
                device=device, clamp_edge_threshold=0.0, bidirectional=bidirectional, dataset_name=dataset_name)
            model.set_adjacency_graph(oracle_graph)
            model_list.append(model)

        initial_model_name = self.model_args['init_model_name']
        if initial_model_name is not None and initial_model_name != 'None':
            init_model = torch.load(os.path.join(ckpt_folder, initial_model_name))
            for i in range(new_model_number):
                model_list[i].load_state_dict(init_model.state_dict(), strict=True)
        return model_list

    def inference(self, data, time):
        concat_input_list = data[0]
        model = self.model

        new_init = model.predict_init_value(concat_input_list)
        new_init_mean, _ = new_init[:, :self.input_size + 1], new_init[:, self.input_size + 1:]
        time = FloatTensor(time).to(self.device)
        new_predict_value = odeint(self.model.derivative, new_init_mean, time)
        new_predict_value_list = chunk(new_predict_value, chunks=self.batch_size, dim=0)
        return new_predict_value_list

    def predict(self, concat_input_list, label_time_list):
        models = self.models
        predict_value_list = []
        for model in models:
            predict_value = model(concat_input_list, label_time_list)
            predict_value_list.append(predict_value)
        return predict_value_list

    def predict_loss(self, predict_value_list, label_feature_list, label_mask_list, label_type_list):
        models = self.models
        loss = 0
        for model, predict_value in zip(models, predict_value_list):
            output_dict = model.loss_calculate(predict_value, label_feature_list, label_mask_list, label_type_list)
            loss += output_dict['loss']
        return loss

    @staticmethod
    def treatment_loss(predict_value_list):
        predict_value_list = squeeze(stack(predict_value_list))
        treatment_loss = cdist(predict_value_list, predict_value_list)
        treatment_loss = treatment_loss.mean()
        return treatment_loss

    def set_mode(self, mode):
        assert mode == 'predict' or mode == 'treatment'
        if mode == 'predict':
            for model in self.models:
                model.derivative.set_treatment(None, None, None)
        else:
            for model in self.models:
                model.derivative.set_treatment(self.treatment_idx, self.treatment_value, self.treatment_time)
