import numpy as np
from default_config import args, logger, ckpt_folder, adjacency_mat_folder, oracle_graph_dict
from model.causal_trajectory_prediction import TrajectoryPrediction
from torch.optim import Adam
from torch import FloatTensor
from util import get_data_loader, save_model, LagrangianMultiplierStateUpdater, predict_performance_evaluation


def train(train_dataloader, val_loader, model, multiplier_updater, optimizer, argument):
    max_epoch = argument['max_epoch']
    max_iteration = argument['max_iteration']
    eval_iter_interval = argument['eval_iter_interval']
    clamp_edge_flag = argument['clamp_edge_flag']
    save_interval = argument['save_iter_interval']
    constraint_type = argument['constraint_type']
    weight = argument['sparse_constraint_weight']
    device = argument['device']

    assert clamp_edge_flag == 'True' or clamp_edge_flag == 'False'
    clamp_edge_flag = True if clamp_edge_flag == 'True' else False

    iter_idx = 0
    predict_performance_evaluation(model, train_dataloader, 'train', 0, 0)
    predict_performance_evaluation(model, val_loader, 'valid', 0, 0)
    model.print_graph(0, adjacency_mat_folder)
    logger.info('--------------------start training--------------------')
    for epoch_idx in range(max_epoch):
        for batch in train_dataloader:
            iter_idx += 1

            # 认为定义lamb和mu的更新频率
            graph_constraint, sparse_constraint = model.calculate_constraint()
            lamb, mu = multiplier_updater.update(model, iter_idx)
            if argument['constraint_type'] == 'DAG':
                constraint = lamb * graph_constraint + 0.5 * mu * graph_constraint**2 + sparse_constraint * weight
            elif argument['constraint_type'] == 'sparse':
                constraint = sparse_constraint * weight
            elif argument['constraint_type'] == 'none':
                constraint = FloatTensor([0]).to(device)
            else:
                raise ValueError('')

            if iter_idx > max_iteration:
                break

            # 删除部分边确保稀疏性，前面几次不做clamp，确保不要一开始因为初始化的原因出什么毛病
            if iter_idx > 20 and clamp_edge_flag:
                model.clamp_edge()

            input_list, label_feature_list, label_time_list = batch[0], batch[4], batch[5]
            label_mask_list, label_type_list = batch[6], batch[7]
            predict_value_list = model(input_list, label_time_list)
            output_dict = model.loss_calculate(predict_value_list, label_feature_list, label_mask_list, label_type_list)
            loss = output_dict['loss']

            if constraint_type == 'DAG':
                loss = loss + constraint
            elif constraint_type == 'sparse':
                loss = loss + constraint
            elif constraint_type == 'none':
                loss = loss
            else:
                raise ValueError('')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_idx % eval_iter_interval == 0:
                # predict_performance_evaluation(model, train_dataloader, 'train', epoch_idx, iter_idx)
                predict_performance_evaluation(model, val_loader, 'valid', epoch_idx, iter_idx)
                model.print_graph(iter_idx, adjacency_mat_folder)
                if iter_idx % save_interval == 0:
                    save_model(model, 'CTP', ckpt_folder, epoch_idx, iter_idx, argument, 'predict')
                    # save_graph(model, 'CTP', adjacency_mat_folder, epoch_idx, iter_idx, argument)
    return model


def get_oracle_causal_graph(prior_causal_mask, name_id_dict):
    assert prior_causal_mask in oracle_graph_dict
    oracle_graph = oracle_graph_dict[prior_causal_mask]
    logger.info('prior causal mask')
    logger.info(oracle_graph)
    bool_graph = np.zeros([len(oracle_graph), len(oracle_graph)])
    new_dict = {key: name_id_dict[key] for key in name_id_dict}
    if 'hidden' not in new_dict and len(name_id_dict) == len(oracle_graph) - 1:
        new_dict['hidden'] = len(new_dict)
    for cause in oracle_graph:
        for consequence in oracle_graph[cause]:
            idx_1, idx_2 = new_dict[cause], new_dict[consequence]
            bool_graph[idx_1, idx_2] = oracle_graph[cause][consequence]
    return bool_graph


def get_data(argument):
    # data setting
    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    mask_tag = argument['mask_tag']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    batch_size = argument['batch_size']
    device = argument['device']
    minimum_observation = argument['minimum_observation']

    dataloader_dict, name_id_dict, oracle_graph, id_type_list, _ = get_data_loader(
        dataset_name, data_path, batch_size, mask_tag, minimum_observation, reconstruct_input, predict_label,
        device=device)

    if len(id_type_list) <= 10:
        logger.info('name id dict')
        logger.info(name_id_dict)
        logger.info('id type list')
        logger.info(id_type_list)
    else:
        logger.info('The oracle graph, id type list, and name id dict are not printed because of high dimension')
    return dataloader_dict, name_id_dict, oracle_graph, id_type_list


def get_model(argument, id_type_list):
    input_size = argument['input_size']
    time_offset = argument['time_offset']
    data_mode = argument['distribution_mode']
    hidden_size = argument['hidden_size']
    bidirectional = argument['init_net_bidirectional']
    batch_first = argument['batch_first']
    dataset_name = argument['dataset_name']
    process_name = argument['process_name']
    non_linear_mode = argument['non_linear_mode']
    device = argument['device']

    # graph setting
    hidden_flag = argument['hidden_flag']
    constraint = argument['constraint_type']
    clamp_edge_threshold = argument['clamp_edge_threshold']


    model = TrajectoryPrediction(
        hidden_flag=hidden_flag, constraint=constraint, input_size=input_size, hidden_size=hidden_size,
        batch_first=batch_first, time_offset=time_offset, input_type_list=id_type_list, non_linear_mode=non_linear_mode,
        device=device, clamp_edge_threshold=clamp_edge_threshold, bidirectional=bidirectional,data_mode=data_mode,
        dataset_name=dataset_name, process_name=process_name)
    return model


def get_lagrangian_updater(argument, validation_dataloader):
    init_lambda = argument['init_lambda_predict']
    init_mu = argument['init_mu_predict']
    eta = argument['eta_predict']
    gamma = argument['gamma_predict']
    lagrangian_converge_threshold = argument['lagrangian_converge_threshold_predict']
    update_window = argument['update_window_predict']
    max_lambda = argument['max_lambda_predict']
    max_mu = argument['max_mu_predict']
    multiplier_updater = LagrangianMultiplierStateUpdater(
        init_lambda=init_lambda, init_mu=init_mu, gamma=gamma, eta=eta, update_window=update_window, max_mu=max_mu,
        dataloader=validation_dataloader, converge_threshold=lagrangian_converge_threshold, max_lambda=max_lambda,
    )
    return multiplier_updater


def framework(argument):
    dataloader_dict, name_id_dict, _, id_type_list = get_data(argument)
    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']
    test_dataloader = dataloader_dict['test']
    prior_causal_mask_name = argument['prior_causal_mask']

    prior_causal_mask = get_oracle_causal_graph(prior_causal_mask_name, name_id_dict)
    model = get_model(argument, id_type_list)
    model.set_adjacency(prior_causal_mask)
    multiplier_updater = get_lagrangian_updater(argument, validation_dataloader)

    optimizer = Adam(model.parameters())
    model = train(train_dataloader, validation_dataloader, model, multiplier_updater, optimizer, argument)
    predict_performance_evaluation(model, test_dataloader, 'test')


if __name__ == '__main__':
    framework(args)
