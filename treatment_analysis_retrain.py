import os
import numpy as np
from default_config import  args, ckpt_folder, oracle_graph_dict
from util import LagrangianMultiplierStateUpdater, get_data_loader
from model.treatment_effect_evaluation import TreatmentEffectEstimator
from torch.optim import Adam
from torch import FloatTensor


def preset_graph_converter(id_dict, graph):
    dag_graph = np.zeros([len(id_dict), len(id_dict)])
    bi_graph = np.zeros([len(id_dict), len(id_dict)])
    for key_1 in graph['dag']:
        for key_2 in graph['dag'][key_1]:
            idx_1, idx_2 = id_dict[key_1], id_dict[key_2]
            dag_graph[idx_1, idx_2] = graph['dag'][key_1][key_2]
    for key_1 in graph['bi']:
        for key_2 in graph['bi'][key_1]:
            idx_1, idx_2 = id_dict[key_1], id_dict[key_2]
            bi_graph[idx_1, idx_2] = graph['bi'][key_1][key_2]
    dag_graph, bi_graph = FloatTensor(dag_graph), FloatTensor(bi_graph)
    return {'dag': dag_graph, 'bi': bi_graph}


def framework(argument, ckpt_name, preset_graph):
    model_ckpt_path = os.path.join(ckpt_folder, ckpt_name)

    # data setting
    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    mask_tag = argument['mask_tag']

    # data loader setting
    input_size = argument['input_size']
    minimum_observation = argument['minimum_observation']

    # treatment analysis
    batch_size = argument['batch_size']
    mode = argument['mode']
    sample_multiplier = argument['sample_multiplier']
    device = argument['device']
    treatment_feature = argument['treatment_feature']
    treatment_time = argument['treatment_time']
    treatment_value = argument['treatment_value']

    # lagrangian
    init_lambda = argument['init_lambda_treatment']
    init_mu = argument['init_mu_treatment']
    eta = argument['eta_treatment']
    gamma = argument['gamma_treatment']
    lagrangian_converge_threshold = argument['lagrangian_converge_threshold_treatment']
    update_window = argument['update_window_treatment']

    dataloader_dict, name_id_dict, _ = \
        get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                        reconstruct_input, predict_label, device=device)
    treatment_idx = name_id_dict[treatment_feature]
    preset_graph = preset_graph_converter(name_id_dict, preset_graph)

    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    model = TreatmentEffectEstimator(
        model_ckpt_path=model_ckpt_path, dataset_name=dataset_name, treatment_idx=treatment_idx,
        treatment_time=treatment_time, treatment_value=treatment_value, device=device, preset_graph=preset_graph,
        mode=mode, sample_multiplier=sample_multiplier, batch_size=batch_size, input_size=input_size)
    multiplier_updater = LagrangianMultiplierStateUpdater(
        init_lambda=init_lambda, init_mu=init_mu, gamma=gamma, eta=eta, update_window=update_window,
        dataloader=validation_dataloader, converge_threshold=lagrangian_converge_threshold)
    optimizer = Adam(model.parameters())

    for batch in train_dataloader:
        output_dict = model.re_fit(batch)
        loss = output_dict


if __name__ == '__main__':
    model_ckpt_name = 'predict.CPA.hao_true.ADMG.ancestral.20230322123827.0.1.model'
    oracle_graph = oracle_graph_dict[model_ckpt_name]
    framework(args, model_ckpt_name, oracle_graph)
