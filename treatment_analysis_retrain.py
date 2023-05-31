import os
import numpy as np
from default_config import  args, ckpt_folder, oracle_graph_dict
from util import LagrangianMultiplierStateUpdater, get_data_loader
from model.treatment_effect_evaluation import TreatmentEffectEstimator
from torch.optim import Adam

def model_refit(train_dataloader, model, multiplier_updater, optimizer):
    # for batch in train_dataloader:
    #     output_dict = model.re_fit(batch)
    return model


def treatment_trajectory_prediction(train_dataloader, model, treatment_idx, treatment_value, treatment_time,
                                    predict_time_list):
    model.set_treatment(treatment_idx, treatment_value, treatment_time)
    for batch in train_dataloader:
        output = model.inference(batch, predict_time_list)
        print('')


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
    treatment_feature_train = argument['treatment_feature_train']
    treatment_time_train = argument['treatment_time_train']
    treatment_value_train = argument['treatment_value_train']
    treatment_feature_eval = argument['treatment_feature_eval']
    treatment_time_eval = argument['treatment_time_eval']
    treatment_value_eval = argument['treatment_value_eval']
    clamp_edge_threshold = argument['treatment_clamp_edge_threshold']
    assert treatment_feature_eval == treatment_feature_train

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
    treatment_idx = name_id_dict[treatment_feature_train]
    preset_graph = preset_graph_converter(name_id_dict, preset_graph)

    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    model = TreatmentEffectEstimator(
        model_ckpt_path=model_ckpt_path, dataset_name=dataset_name, treatment_idx=treatment_idx,
        treatment_time=treatment_time_train, treatment_value=treatment_value_train, device=device,
        preset_graph=preset_graph, mode=mode, sample_multiplier=sample_multiplier, batch_size=batch_size,
        input_size=input_size, clamp_edge_threshold=clamp_edge_threshold)
    multiplier_updater = LagrangianMultiplierStateUpdater(
        init_lambda=init_lambda, init_mu=init_mu, gamma=gamma, eta=eta, update_window=update_window,
        dataloader=validation_dataloader, converge_threshold=lagrangian_converge_threshold)
    optimizer = Adam(model.parameters())

    re_fit_flag = model.re_fit_flag
    if re_fit_flag:
        model = model_refit(train_dataloader, model, multiplier_updater, optimizer)

    predict_time_list = [i for i in range(51, 65)]
    treatment_trajectory_prediction(train_dataloader, model, treatment_idx, treatment_value_eval, treatment_time_eval,
                                    predict_time_list)


if __name__ == '__main__':
    model_ckpt_name = 'predict.CPA.hao_true.ADMG.ancestral.20230323050007.74.3000.model'
    oracle_graph = oracle_graph_dict[model_ckpt_name]
    framework(args, model_ckpt_name, oracle_graph)
