import copy
import numpy as np
import torch
from torch import no_grad, FloatTensor, mean
from default_config import args, logger, ckpt_folder, oracle_graph_dict
from util import get_data_loader, save_model
from model.treatment_effect_evaluation import TreatmentEffectEstimator
from torch.optim import Adam
import random


def train(train_loader, val_loader, model, optimizer_predict, optimizer_treatment, max_epoch, max_iteration,
          eval_iter_interval, observation_time, treatment_warm_iter, random_observation_time, argument):
    iter_idx = 0
    flag = True
    for epoch_idx in range(max_epoch):
        for batch in train_loader:
            input_list, label_feature_list, label_time_list = batch[0], batch[4], batch[5]
            label_mask_list, label_type_list = batch[6], batch[7]
            if iter_idx > max_iteration:
                break

            if iter_idx % eval_iter_interval == 0:
                observation_time_list = [FloatTensor([observation_time]) for _ in range(len(input_list))]
                # performance_evaluation(model, train_loader, 'train', observation_time_list, epoch_idx, iter_idx)
                performance_evaluation(model, val_loader, 'val', observation_time_list, epoch_idx, iter_idx)
                save_model(model, 'TEP', ckpt_folder, epoch_idx, iter_idx, argument, 'treatment')

            # set observation time
            if random_observation_time:
                train_time = random.uniform(model.treatment_time, observation_time)
            else:
                train_time = observation_time
            train_observe_time_list = [FloatTensor([train_time]) for _ in range(len(input_list))]

            if flag:
                model.set_mode('predict')
                predict_list = model.predict(input_list, label_time_list)
                loss = model.predict_loss(predict_list, label_feature_list, label_mask_list, label_type_list)
                optimizer_predict.zero_grad()
                loss.backward()
                optimizer_predict.step()
            else:
                model.set_mode('treatment')
                optimize_method = argument['treatment_optimize_method']
                predict_list = model.predict(input_list, train_observe_time_list)
                if optimize_method == 'max' or optimize_method == 'difference':
                    loss = -1 * model.treatment_loss(predict_list)
                else:
                    loss = model.treatment_loss(predict_list)
                optimizer_treatment.zero_grad()
                loss.backward()
                optimizer_treatment.step()

            if treatment_warm_iter == iter_idx:
                filter_set = remove_module(model, val_loader, argument['treatment_filter_threshold'])
                model.filter_set = model.filter_set.union(filter_set)

            if treatment_warm_iter < iter_idx:
                flag = not flag
            iter_idx += 1
    logger.info('optimization finished')
    return model


def remove_module(model, val_loader, threshold):
    with torch.no_grad():
        models = model.models
        filter_set = set()
        for i, model in enumerate(models):
            loss, idx = 0, 0
            for batch in val_loader:
                input_list, label_feature_list, label_time_list = batch[0], batch[4], batch[5]
                label_mask_list, label_type_list = batch[6], batch[7]
                predict_value_list = model(input_list, label_time_list)
                output_dict = model.loss_calculate(predict_value_list, label_feature_list, label_mask_list,
                                                   label_type_list)
                loss = output_dict['loss'] + loss
                idx = idx + 1
            if loss / idx > threshold:
                if len(filter_set) < len(models) - 2:
                    filter_set.add(i)
    logger.info('{} models are filtered after warming and {} remains'.
                format(len(filter_set), len(models)-len(filter_set)))
    return filter_set


def performance_evaluation(model, loader, loader_fraction, observation_time_list, epoch_idx=None, iter_idx=None):
    with no_grad():
        pred_loss_list, treatment_loss_list = [], []
        for batch in loader:
            input_list, label_feature_list, label_time_list = batch[0], batch[4], batch[5]
            label_mask_list, label_type_list = batch[6], batch[7]
            model.set_mode('predict')
            predict_list = model.predict(input_list, label_time_list)
            pred_loss = model.predict_loss(predict_list, label_feature_list, label_mask_list, label_type_list)
            model.set_mode('treatment')
            predict_list = model.predict(input_list, observation_time_list)
            treatment_loss = model.treatment_loss(predict_list)
            pred_loss_list.append(pred_loss)
            treatment_loss_list.append(treatment_loss)

        pred_loss = mean(FloatTensor(pred_loss_list)).item()
        treatment_loss = mean(FloatTensor(treatment_loss_list)).item()
    if epoch_idx is not None:
        logger.info('epoch: {:>4d}, iter: {:>4d}, {:>6s} predict loss: {:>8.8f}, treatment loss: {:>8.8f}'
            .format(epoch_idx, iter_idx, loader_fraction, pred_loss, treatment_loss))
    else:
        logger.info('final {}, predict loss: {:>8.8f}, treatment loss: {:>8.8f}'
                    .format(loader_fraction, pred_loss, treatment_loss))


def framework(argument, oracle_graph):
    # data setting
    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    random_observation_time = True if argument['treatment_random_observation_time'] else False
    mask_tag = argument['mask_tag']

    # data loader setting
    minimum_observation = argument['minimum_observation']

    # treatment analysis
    batch_size = argument['batch_size']
    device = argument['device']

    treatment_warm_iter = argument['treatment_warm_iter']
    treatment_predict_lr = argument['treatment_predict_lr']
    treatment_treatment_lr = argument['treatment_treatment_lr']
    treatment_feature = argument['treatment_feature']
    treatment_time = argument['treatment_time']
    treatment_observation_time = argument['treatment_observation_time']
    treatment_value = argument['treatment_value']
    max_epoch = argument['treatment_max_epoch']
    max_iter = argument['treatment_max_iter']
    eval_iter_interval = argument['treatment_eval_iter_interval']
    new_model_number = argument['treatment_new_model_number']
    process_name = argument['process_name']
    treatment_optimize_method = argument['treatment_optimize_method']

    model_args = {
        'init_model_name': argument['treatment_init_model_name'],
        'hidden_flag': argument['hidden_flag'],
        'input_size': argument['input_size'],
        'distribution_mode': argument['distribution_mode'],
        'batch_first': argument['batch_first'],
        'hidden_size': argument['hidden_size'],
        'bidirectional': argument['init_net_bidirectional'],
        'device': argument['device'],
        'dataset_name': argument['dataset_name'],
        'time_offset': argument['time_offset'],
        'non_linear_mode': argument['non_linear_mode']
    }


    dataloader_dict, name_id_dict, _, id_type_list, stat_dict = \
        get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                        reconstruct_input, predict_label, device=device)

    # 干预经过正态转换
    treatment_value = (treatment_value - stat_dict[treatment_feature][0]) / stat_dict[treatment_feature][1]
    treatment_idx = name_id_dict[treatment_feature]
    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']

    oracle_graph = convert_oracle_graph(oracle_graph, name_id_dict)

    models = TreatmentEffectEstimator(
        dataset_name=dataset_name, device=device, treatment_idx=treatment_idx, oracle_graph=oracle_graph,
        batch_size=batch_size, treatment_feature=treatment_feature, new_model_number=new_model_number,
        id_type_list=id_type_list, model_args=model_args, treatment_time=treatment_time,
        treatment_value=treatment_value, process_name=process_name, optimize_method=treatment_optimize_method
    )

    para_list_1, para_list_2 = [], []
    for model in models.models:
        for para in model.parameters():
            para_list_1.append(para)
            para_list_2.append(para)
    optimizer_predict = Adam(para_list_1, lr=treatment_predict_lr)
    optimizer_treatment = Adam(para_list_2, lr=treatment_treatment_lr)
    train(train_dataloader, validation_dataloader, models, optimizer_predict, optimizer_treatment,
          max_epoch, max_iter, eval_iter_interval, treatment_observation_time, treatment_warm_iter,
          random_observation_time, argument)


def convert_oracle_graph(oracle_graph, name_id_dict):
    new_name_id_dict = copy.deepcopy(name_id_dict)
    np_oracle = np.zeros([len(oracle_graph), len(oracle_graph)])
    if 'hidden' in oracle_graph:
        new_name_id_dict['hidden'] = len(new_name_id_dict)
    for key_1 in oracle_graph:
        for key_2 in oracle_graph[key_1]:
            value = oracle_graph[key_1][key_2]
            idx_1 = new_name_id_dict[key_1]
            idx_2 = new_name_id_dict[key_2]
            np_oracle[idx_1, idx_2] = value
    converted_graph = FloatTensor(np_oracle)
    return converted_graph


def read_oracle_graph(dataset_name, true_casual):
    if dataset_name == 'hao_true_lmci' and true_casual:
        return oracle_graph_dict['hao_true_causal']
    elif dataset_name == 'hao_true_lmci' and not true_casual:
        return oracle_graph_dict['hao_true_not_causal']
    else:
        raise ValueError('')


def main():
    dataset_name = args['dataset_name']
    true_causal = True if args['treatment_true_causal'] == "True" else False
    oracle_graph = read_oracle_graph(dataset_name, true_causal)
    framework(args, oracle_graph)


if __name__ == '__main__':
    main()
