import os
from default_config import  args, ckpt_folder
from util import LagrangianMultiplierStateUpdater, get_data_loader
from model.treatment_effect_evaluation import TreatmentEffectEstimator
from torch.optim import Adam



def framework(argument, ckpt_name):
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

    dataloader_dict, name_id_dict, oracle_graph = \
        get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                        reconstruct_input, predict_label, device=device)
    treatment_idx = name_id_dict[treatment_feature]

    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    model = TreatmentEffectEstimator(
        model_ckpt_path=model_ckpt_path, dataset_name=dataset_name, treatment_idx=treatment_idx,
        treatment_time=treatment_time, treatment_value=treatment_value, device=device,
        mode=mode, sample_multiplier=sample_multiplier, batch_size=batch_size, input_size=input_size)
    multiplier_updater = LagrangianMultiplierStateUpdater(
        init_lambda=init_lambda, init_mu=init_mu, gamma=gamma, eta=eta, update_window=update_window,
        dataloader=validation_dataloader, converge_threshold=lagrangian_converge_threshold)
    optimizer = Adam(model.parameters())

    for batch in train_dataloader:
        output_dict = model.re_fit(batch)
        loss = output_dict


if __name__ == '__main__':
    model_ckpt_name = 'CPA.hao_false.DAG.default.20230407075131.0.1.model'
    framework(args, model_ckpt_name)
