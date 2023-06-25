import os
import numpy as np
from torch import load, no_grad, FloatTensor, stack, squeeze, permute
from default_config import args, ckpt_folder, oracle_graph_dict, logger, treatment_result_folder
from util import get_data_loader, preset_graph_converter, get_oracle_model
from model.treatment_effect_evaluation import TreatmentEffectEstimator
from torch.optim import Adam
import pickle


def model_refit(train_loader, val_loader, model, optimizer, max_epoch, max_iteration, converge_threshold,
                eval_iter_interval, treatment_idx, treatment_value, treatment_time):
    iter_idx = 0
    previous_loss = 0
    for epoch_idx in range(max_epoch):
        for batch in train_loader:
            iter_idx += 1
            if iter_idx > max_iteration:
                break
            input_list, time_list = batch[0], batch[5]
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
                        input_list = val_batch[0]
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


def treatment_trajectory_prediction(data, model, treatment_feature, treatment_idx, treatment_value, treatment_time,
                                    predict_time_list, model_ckpt_name, stat_dict):
    model.set_treatment(treatment_idx, treatment_value, treatment_time)
    output = model.inference(data, predict_time_list)
    time_offset = model.trained_model.time_offset
    result_list = []
    id_list, init_list, para_list = data[10], data[11], data[12]
    for init, para in zip(init_list, para_list):
        oracle_model = get_oracle_model(model_ckpt_name, init, para, time_offset, stat_dict)
        oracle_model.set_treatment(treatment_feature, treatment_time, treatment_value)
        result_list.append(oracle_model.inference(predict_time_list))

    output = squeeze(stack(output, dim=0))
    output = permute(output, (1, 0, 2)).detach().to('cpu').numpy()

    oracle_result = np.stack(output, axis=0)
    return {'oracle_result': oracle_result, 'predict_result': output, 'treatment_time': treatment_time,
            'predict_time_list': predict_time_list, 'id_list': id_list}


def framework(argument, trained_model, model_ckpt_name, preset_graph):
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

    dataloader_dict, name_id_dict, _, id_type_list, stat_dict = \
        get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                        reconstruct_input, predict_label, device=device)
    treatment_idx = name_id_dict[treatment_feature]
    preset_graph = preset_graph_converter(name_id_dict, preset_graph)

    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    model = TreatmentEffectEstimator(
        trained_model=trained_model, dataset_name=dataset_name, device=device, treatment_idx=treatment_idx,
        preset_graph=preset_graph, mode=mode, sample_multiplier=sample_multiplier, batch_size=batch_size,
        input_size=input_size, clamp_edge_threshold=clamp_edge_threshold, time_offset=time_offset,
        input_type_list=id_type_list, treatment_feature=treatment_feature)
    optimizer = Adam(model.parameters())

    re_fit_flag = model.re_fit_flag
    if re_fit_flag:
        treatment_time = FloatTensor([treatment_time]).to(device)
        model = model_refit(train_dataloader, validation_dataloader, model, optimizer, max_epoch, max_iter,
                            treatment_refit_converge_threshold, eval_iter_interval, treatment_idx, treatment_value,
                            treatment_time)

    result_list = []
    for batch in validation_dataloader:
        predict_time_list = FloatTensor(np.array([i for i in range(51, 65)]))
        result = treatment_trajectory_prediction(
            batch, model, treatment_feature, treatment_idx, treatment_value, treatment_time,
            predict_time_list, model_ckpt_name, stat_dict)
        result_list.append(result)
    return result_list


def main():
    model_ckpt_name = 'predict.CPA.hao_true.ADMG.ancestral.20230330041633.32.1300.model'
    treatment_result_path = os.path.join(treatment_result_folder, model_ckpt_name)
    model_ckpt_path = os.path.join(ckpt_folder, model_ckpt_name)
    model = load(model_ckpt_path)

    oracle_graph = oracle_graph_dict[model_ckpt_name]
    result_list = framework(args, model, model_ckpt_name, oracle_graph)
    pickle.dump(result_list, open(treatment_result_path, 'wb'))

if __name__ == '__main__':
    main()
