from default_config import args, logger, ckpt_folder, adjacency_mat_folder
from model.causal_trajectory_prediction import TrajectoryPrediction
from torch.optim import Adam
from util import get_data_loader, save_model, LagrangianMultiplierStateUpdater, predict_performance_evaluation


def train(train_dataloader, val_loader, model, multiplier_updater, optimizer, argument):
    max_epoch = argument['max_epoch']
    max_iteration = argument['max_iteration']
    model_converge_threshold = argument['model_converge_threshold']
    eval_iter_interval = argument['eval_iter_interval']
    clamp_edge_flag = argument['clamp_edge_flag']
    save_interval = argument['save_iter_interval']
    causal_derivative_flag = argument['causal_derivative_flag']
    assert clamp_edge_flag == 'True' or clamp_edge_flag == 'False'
    clamp_edge_flag = True if clamp_edge_flag == 'True' else False
    causal_derivative_flag = True if causal_derivative_flag == "True" else False

    iter_idx = 0
    predict_performance_evaluation(model, train_dataloader, 'train', 0, 0)
    predict_performance_evaluation(model, val_loader, 'valid', 0, 0)

    model.dump_graph(0, adjacency_mat_folder)
    logger.info('--------------------start training--------------------')
    for epoch_idx in range(max_epoch):
        for batch in train_dataloader:
            iter_idx += 1
            if iter_idx > max_iteration:
                break
            lamb, mu = multiplier_updater.update(model, iter_idx)
            constraint = model.calculate_constraint()

            if causal_derivative_flag:
                if constraint < model_converge_threshold:
                    return model
                # 删除部分边确保稀疏性，前面几次不做clamp，确保不要一开始因为初始化的原因出什么毛病
                if iter_idx > 20 and clamp_edge_flag:
                    model.clamp_edge()

            input_list, label_feature_list, label_time_list = batch[0], batch[4], batch[5]
            label_mask_list, label_type_list = batch[6], batch[7]
            predict_value_list = model(input_list, label_time_list)
            output_dict = model.loss_calculate(predict_value_list, label_feature_list, label_mask_list, label_type_list)
            loss = output_dict['loss']
            loss = loss + lamb * constraint + 1 / 2 * mu * constraint ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_idx % eval_iter_interval == 0:
                predict_performance_evaluation(model, train_dataloader, 'train', epoch_idx, iter_idx)
                predict_performance_evaluation(model, val_loader, 'valid', epoch_idx, iter_idx)
                model.dump_graph(iter_idx, adjacency_mat_folder)

            if iter_idx == 1 or iter_idx % save_interval == 0:
                save_model(model, 'CTP', ckpt_folder, epoch_idx, iter_idx, argument, 'predict')
    return model


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
        logger.info('oracle')
        logger.info(oracle_graph)
        logger.info('id type list')
        logger.info(id_type_list)
    else:
        logger.info('The oracle graph, id type list, and name id dict are not printed because of high dimension')
    return dataloader_dict, name_id_dict, oracle_graph, id_type_list


def get_model(argument, id_type_list):
    input_size = argument['input_size']
    time_offset = argument['time_offset']
    mode = argument['distribution_mode']
    hidden_size = argument['hidden_size']
    mediate_size = argument['mediate_size']
    bidirectional = argument['init_net_bidirectional']
    batch_first = True if argument['batch_first'] == 'True' else False
    dataset_name = argument['dataset_name']
    device = argument['device']

    # graph setting
    graph_type = argument['graph_type']
    constraint = argument['constraint_type']
    clamp_edge_threshold = argument['clamp_edge_threshold']

    causal_derivative_flag = argument['causal_derivative_flag']

    model = TrajectoryPrediction(
        graph_type=graph_type, constraint=constraint, input_size=input_size, hidden_size=hidden_size, mode=mode,
        batch_first=batch_first, mediate_size=mediate_size, time_offset=time_offset, input_type_list=id_type_list,
        device=device, clamp_edge_threshold=clamp_edge_threshold, bidirectional=bidirectional,
        dataset_name=dataset_name, causal_derivative_flag=causal_derivative_flag)
    return model


def get_lagrangian_updater(argument, validation_dataloader):
    init_lambda = argument['init_lambda_predict']
    init_mu = argument['init_mu_predict']
    eta = argument['eta_predict']
    gamma = argument['gamma_predict']
    lagrangian_converge_threshold = argument['lagrangian_converge_threshold_predict']
    update_window = argument['update_window_predict']
    multiplier_updater = LagrangianMultiplierStateUpdater(
        init_lambda=init_lambda, init_mu=init_mu, gamma=gamma, eta=eta, update_window=update_window,
        dataloader=validation_dataloader, converge_threshold=lagrangian_converge_threshold)
    return multiplier_updater


def framework(argument):
    dataloader_dict, name_id_dict, oracle_graph, id_type_list = get_data(argument)
    train_dataloader = dataloader_dict['train']
    validation_dataloader = dataloader_dict['valid']
    test_dataloader = dataloader_dict['test']

    model = get_model(argument, id_type_list)
    multiplier_updater = get_lagrangian_updater(argument, validation_dataloader)

    optimizer = Adam(model.parameters())
    model = train(train_dataloader, validation_dataloader, model, multiplier_updater, optimizer, argument)
    predict_performance_evaluation(model, test_dataloader, 'test')


if __name__ == '__main__':
    framework(args)
