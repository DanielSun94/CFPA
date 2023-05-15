from default_config import  args
from util import LagrangianMultiplierStateUpdater, get_data_loader
from model.treatment_effect_evaluation import TreatmentEffectEstimator
from torch.optim import Adam


def re_train():
    print('')


def framework(argument):
    # data setting
    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    mask_tag = argument['mask_tag']

    # data loader setting
    batch_first = True if argument['batch_first'] == 'True' else False
    minimum_observation = argument['minimum_observation']

    # treatment analysis
    model_ckpt_path = argument['model_ckpt_path']
    batch_size = argument['batch_size']
    device = argument['device']
    treatment_feature = argument['treatment_feature']
    treatment_time = argument['treatment_time']
    treatment_value = argument['treatment_value']
    oracle_graph_flag = True if argument['oracle_graph_flag'] == 'True' else False

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
    if not oracle_graph_flag:
        oracle_graph = None

    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    model = TreatmentEffectEstimator(model_ckpt_path, dataset_name, treatment_feature, treatment_time, treatment_value,
                                     device, name_id_dict, oracle_graph)
    multiplier_updater = LagrangianMultiplierStateUpdater(
        init_lambda=init_lambda, init_mu=init_mu, gamma=gamma, eta=eta, update_window=update_window,
        dataloader=validation_dataloader, converge_threshold=lagrangian_converge_threshold)
    optimizer = Adam(model.parameters())

    print('')


if __name__ == '__main__':
    framework(args)
