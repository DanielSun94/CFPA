import torch
import os
from default_config import ckpt_folder
from model.causal_trajectory_prediction import TrajectoryPrediction
from torch import stack, squeeze, cdist, permute
from torch.nn import Module
from torch.nn import ModuleList


class TreatmentEffectEstimator(Module):
    def __init__(self, dataset_name, device, batch_size, new_model_number, oracle_graph, treatment_idx,
                 treatment_value, treatment_time, treatment_feature, id_type_list, process_name, model_args,
                 optimize_method):
        super().__init__()

        # treatment time为-1，-2时表示施加干预的时间是最后一次入院的时间/倒数第二次入院的时间，是正数时代表任意指定时间
        assert optimize_method == 'max' or optimize_method == 'difference' or optimize_method == 'mim'
        if optimize_method == 'max' or optimize_method == 'min':
            assert new_model_number == 1 or new_model_number is None

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
        self.optimize_method = optimize_method
        self.process_name = process_name
        self.filter_set = set()
        self.models = self.build_models(new_model_number, oracle_graph)

    def build_models(self, new_model_number, oracle_graph):
        model_list = ModuleList()
        hidden_flag = self.model_args['hidden_flag']
        input_size = self.model_args['input_size']
        data_mode = self.model_args['distribution_mode']
        hidden_size = self.model_args['hidden_size']
        batch_first = self.model_args['batch_first']
        bidirectional = self.model_args['bidirectional']
        device = self.model_args['device']
        non_linear_mode = self.model_args['non_linear_mode']
        dataset_name = self.model_args['dataset_name']


        time_offset = self.model_args['time_offset']
        id_type_list = self.id_type_list

        for i in range(new_model_number):
            model = TrajectoryPrediction(
                hidden_flag=hidden_flag, constraint='none', input_size=input_size, hidden_size=hidden_size,
                data_mode=data_mode, batch_first=batch_first, time_offset=time_offset, input_type_list=id_type_list,
                device=device, clamp_edge_threshold=0.0, bidirectional=bidirectional, dataset_name=dataset_name,
                process_name=self.process_name, non_linear_mode=non_linear_mode
            )
            model.set_adjacency_graph(oracle_graph)
            model_list.append(model)

        initial_model_name = self.model_args['init_model_name']
        if initial_model_name is not None and initial_model_name != 'None':
            init_model = torch.load(os.path.join(ckpt_folder, initial_model_name))
            for i in range(new_model_number):
                model_list[i].load_state_dict(init_model, strict=True)
        return model_list.to(device)

    def predict(self, concat_input_list, label_time_list):
        models = self.models
        predict_value_list = []
        for i, model in enumerate(models):
            if i in self.filter_set:
                continue
            predict_value = model(concat_input_list, label_time_list)
            predict_value_list.append(predict_value)
        return predict_value_list

    def predict_loss(self, predict_value_list, label_feature_list, label_mask_list, label_type_list):
        models = self.models
        loss = 0
        for i, (model, predict_value) in enumerate(zip(models, predict_value_list)):
            if i in self.filter_set:
                continue
            output_dict = model.loss_calculate(predict_value, label_feature_list, label_mask_list, label_type_list)
            loss = output_dict['loss'] + loss
        loss = loss / len(models)
        return loss

    def treatment_loss(self, predict_value_list):
        optimize_type = self.optimize_method
        assert optimize_type == 'max' or optimize_type == 'difference'
        if optimize_type == 'difference':
            predict_value_list = squeeze(stack(predict_value_list), dim=2)
            predict_value_list = permute(predict_value_list, [1, 0, 2])
            treatment_loss = cdist(predict_value_list, predict_value_list)
            treatment_loss = treatment_loss.mean()
        else:
            # 由于实验设置，此处只有一个模型
            assert len(predict_value_list) == 1
            treatment_idx = self.treatment_idx
            target_value = predict_value_list[0][:, 0, treatment_idx]
            treatment_loss = target_value.mean()
        return treatment_loss

    def set_sample_multiplier(self, sample_multiplier):
        assert len(self.models) == 1
        self.models[0].set_sample_multiplier(sample_multiplier)

    def set_mode(self, mode):
        assert mode == 'predict' or mode == 'treatment'
        if mode == 'predict':
            for model in self.models:
                model.derivative.set_treatment(None, None, None)
        else:
            for model in self.models:
                time_offset = model.time_offset
                time = self.treatment_time-time_offset
                model.derivative.set_treatment(self.treatment_idx, self.treatment_value, time)
