import os
import numpy as np
from torch import load, no_grad, FloatTensor, unsqueeze
from default_config import args, ckpt_folder, oracle_graph_dict, logger
from util import get_data_loader, preset_graph_converter
from model.treatment_effect_evaluation import TreatmentEffectEstimator
from torch.optim import Adam


def model_refit(train_loader, val_loader, model, optimizer, max_epoch, max_iteration, converge_threshold,
                eval_iter_interval, treatment_time):
    iter_idx = 0
    previous_loss = 0
    for epoch_idx in range(max_epoch):
        for batch in train_loader:
            iter_idx += 1
            if iter_idx > max_iteration:
                break
            input_list, _, _, _, _, time_list, _, _, _, _ = batch
            time = model.get_predict_time(time_list)
            loss = model.re_fit(input_list, time)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_idx % eval_iter_interval == 0:
                with no_grad():
                    loss, num = 0, 0
                    for val_batch in val_loader:
                        num += 1
                        input_list, _, _, _, _, _, _, _, _, _ = val_batch
                        loss = model.re_fit(input_list, treatment_time) + loss
                    loss = loss / num
                logger.info('epoch: {}, iter: {}, val loss: {}'
                            .format(epoch_idx, iter_idx, loss.detach().to('cpu').item()))
                if previous_loss == 0:
                    continue
                else:
                    if abs(loss-previous_loss) < converge_threshold:
                        logger.info('loss converges')
                        return model
                    else:
                        previous_loss = loss.detach()
    logger.info('optimization finished')
    return model


def treatment_trajectory_prediction(data, model, treatment_idx, treatment_value, treatment_time,
                                    predict_time_list):
    model.set_treatment(treatment_idx, treatment_value, treatment_time)
    output = model.inference(data, predict_time_list)
    print(output)


def framework(argument, ckpt_name, preset_graph):
    model_ckpt_path = os.path.join(ckpt_folder, ckpt_name)

    # data setting
    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    mask_tag = argument['mask_tag']
    time_offset = argument['time_offset']

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
    max_iter = argument['treatment_refit_max_iter']
    max_epoch = argument['treatment_refit_max_epoch']
    treatment_refit_converge_threshold = argument['treatment_refit_converge_threshold']
    eval_iter_interval = argument['treatment_eval_iter_interval']
    clamp_edge_threshold = argument['treatment_clamp_edge_threshold']

    dataloader_dict, name_id_dict, _ = \
        get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                        reconstruct_input, predict_label, device=device)
    treatment_idx = name_id_dict[treatment_feature]
    preset_graph = preset_graph_converter(name_id_dict, preset_graph)

    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    trained_model = load(model_ckpt_path)

    model = TreatmentEffectEstimator(
        trained_model=trained_model, dataset_name=dataset_name, device=device, treatment_idx=treatment_idx,
        preset_graph=preset_graph, mode=mode, sample_multiplier=sample_multiplier, batch_size=batch_size,
        input_size=input_size, clamp_edge_threshold=clamp_edge_threshold, time_offset=time_offset)
    optimizer = Adam(model.parameters())

    re_fit_flag = model.re_fit_flag
    if re_fit_flag:
        treatment_time = FloatTensor([treatment_time]).to(device)
        model = model_refit(train_dataloader, validation_dataloader, model, optimizer, max_epoch, max_iter,
                            treatment_refit_converge_threshold, eval_iter_interval, treatment_time)

    for batch in validation_dataloader:
        predict_time_list = FloatTensor(np.array([i for i in range(51, 65)]))
        treatment_trajectory_prediction(batch, model, treatment_idx, treatment_value, treatment_time,
                                        predict_time_list)


if __name__ == '__main__':
    model_ckpt_name = 'predict.CPA.hao_true.ADMG.ancestral.20230323050007.74.3000.model'
    oracle_graph = oracle_graph_dict[model_ckpt_name]
    framework(args, model_ckpt_name, oracle_graph)
