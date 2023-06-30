import os.path
import numpy as np
import torch
from default_config import ckpt_folder, args, treatment_result_inference_folder
from util import get_data_loader, OracleHaoModel
import pickle
from model.treatment_effect_evaluation import TreatmentEffectEstimator
from treatment_analysis_retrain import convert_oracle_graph, read_oracle_graph

def main(argument):
    assert argument['hidden_flag'] == "True" or argument['hidden_flag'] == "False"
    assert argument['reconstruct_input'] == "True" or argument['reconstruct_input'] == "False"
    assert argument['predict_label'] == "True" or argument['predict_label'] == "False"

    device = argument['device']
    hidden_flag = True if argument['hidden_flag'] == 'True' else False
    mask_tag = argument['mask_tag']
    minimum_observation = argument['minimum_observation']
    batch_size = argument['batch_size']
    dataset_name = argument['dataset_name']
    time_offset = argument['time_offset']
    data_path = argument['data_path']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    t_feature = argument['treatment_feature']
    t_time = argument['treatment_time']
    obs_time = argument['treatment_observation_time']
    t_value = argument['treatment_value']
    constraint = argument['constraint_type']
    inference_model_name = argument['inference_model_name']
    new_model_number = argument['treatment_new_model_number']

    oracle_graph = read_oracle_graph(dataset_name, hidden_flag, constraint)
    dataloader_dict, name_id_dict, _, id_type_list, stat_dict = \
        get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                        reconstruct_input, predict_label, device=device)
    test_dataset = dataloader_dict['test']
    t_idx = name_id_dict[t_feature]
    oracle_graph = convert_oracle_graph(oracle_graph, name_id_dict)

    # 干预要经过正态转换
    mean, std = stat_dict[t_feature]
    t_value = (t_value - mean) / std
    time_list = np.array([(i + 1) * 0.05 * (obs_time - time_offset) + time_offset for i in range(20)])

    model_treatment_dataset = generate_model_behavior(
        hidden_flag, test_dataset, dataset_name, t_feature, t_time, time_list, t_value, t_idx, time_offset,
        inference_model_name, oracle_graph, id_type_list, name_id_dict, device, argument
    )
    model_origin_dataset = generate_model_behavior(
        hidden_flag, test_dataset, dataset_name, None, None, time_list, None, None,time_offset,
        inference_model_name, oracle_graph, id_type_list, name_id_dict, device, argument
    )
    oracle_original_dataset = generate_oracle_behavior(
        hidden_flag, test_dataset, dataset_name, stat_dict, None, None, time_list, None, time_offset)
    oracle_treatment_dataset = generate_oracle_behavior(
        hidden_flag, test_dataset, dataset_name, stat_dict, t_feature, t_time, time_list, t_value, time_offset)


    data_dict = {
        'oracle_origin': oracle_original_dataset,
        'oracle_treatment': oracle_treatment_dataset,
        'model_treatment': model_treatment_dataset,
        'model_origin': model_origin_dataset
    }
    fused_dict = fuse_result(data_dict, time_list)

    file_name = "{}.{}.{}.{}.{}.{}.{}.pkl"\
        .format(dataset_name, hidden_flag, t_feature, t_time, t_value, constraint, new_model_number)
    pickle.dump([fused_dict, time_list], open(os.path.join(treatment_result_inference_folder, file_name), 'wb'))


def fuse_result(data_dict, time_list):
    assert len(data_dict['oracle_origin'][0]) == len(data_dict['oracle_origin'][1])
    assert len(data_dict['oracle_treatment'][0]) == len(data_dict['oracle_treatment'][1])
    assert len(data_dict['model_treatment'][0]) == len(data_dict['model_treatment'][1])
    assert len(data_dict['model_origin'][0]) == len(data_dict['model_origin'][1])
    assert len(data_dict['model_origin'][0]) == len(data_dict['model_treatment'][0])
    result_dict = dict()
    for i in range(len(data_dict['oracle_origin'][1])):
        assert data_dict['oracle_origin'][1][i] not in result_dict
        result_dict[data_dict['oracle_origin'][1][i]] = {'oracle_origin': data_dict['oracle_origin'][0][i]}
    for i in range(len(data_dict['oracle_origin'][1])):
        assert data_dict['oracle_treatment'][1][i] in result_dict
        result_dict[data_dict['oracle_treatment'][1][i]]['oracle_treatment'] = data_dict['oracle_treatment'][0][i]
        assert data_dict['model_treatment'][1][i] in result_dict
        result_dict[data_dict['model_treatment'][1][i]]['model_treatment'] = data_dict['model_treatment'][0][i]
        assert data_dict['model_origin'][1][i] in result_dict
        result_dict[data_dict['model_origin'][1][i]]['model_origin'] = data_dict['model_origin'][0][i]
    return result_dict, time_list


def get_oracle_model(use_hidden, model_name, para_dict, init_dict, time_offset, stat_dict):
    if 'hao' in model_name:
        return OracleHaoModel(use_hidden, para_dict, init_dict, time_offset, stat_dict)
    else:
        raise ValueError('')


def generate_oracle_behavior(hidden_flag, dataloader, dataset, stat_dict, treatment_feature, treatment_time,
                             time_list, treatment_value, time_offset):
    prediction_list, sample_ids = [], []
    for batch in dataloader:
        _, _, _, _, _, _, _, _, _, _, sample_id_list, init_list, para_list = batch
        for sample_id, init, para in zip(sample_id_list, init_list, para_list):
            model = get_oracle_model(hidden_flag, dataset, para, init, time_offset, stat_dict)
            model.set_treatment(treatment_feature, treatment_time, treatment_value)
            prediction = model.inference(time_list)
            prediction_list.append(prediction)
            sample_ids.append(sample_id)
    return prediction_list, sample_ids


def generate_model_behavior(hidden_flag, dataloader, dataset_name, treatment_feature, treatment_time, time_list,
                            treatment_value, treatment_idx, time_offset, inference_model_name, oracle_graph,
                            id_type_list, name_id_dict, device, argument):
    batch_size = dataloader.batch_size
    new_model_number = argument['treatment_new_model_number']
    process_name = argument['process_name']
    model_args = {
        'init_model_name': None,
        'hidden_flag': argument['hidden_flag'],
        'input_size': argument['input_size'],
        'distribution_mode': argument['distribution_mode'],
        'batch_first': argument['batch_first'],
        'hidden_size': argument['hidden_size'],
        'bidirectional': argument['init_net_bidirectional'],
        'device': argument['device'],
        'dataset_name': argument['dataset_name'],
        'time_offset': argument['time_offset']
    }

    model = TreatmentEffectEstimator(
        dataset_name=dataset_name, device=device, treatment_idx=treatment_idx, oracle_graph=oracle_graph,
        batch_size=batch_size, treatment_feature=treatment_feature, new_model_number=new_model_number,
        id_type_list=id_type_list, model_args=model_args, treatment_time=treatment_time,
        process_name=process_name, treatment_value=treatment_value
    )
    trained_model = torch.load(os.path.join(ckpt_folder, inference_model_name))
    model.oracle_graph = trained_model.oracle_graph
    model.load_state_dict(trained_model.state_dict())
    model.to(device)

    assert argument['hidden_flag'] == 'True' or argument['hidden_flag'] == "False"
    assert True if argument['hidden_flag'] == "True" else False == hidden_flag
    assert trained_model.new_model_number == model.new_model_number
    assert trained_model.models[0].hidden_flag == model.models[0].hidden_flag
    assert trained_model.models[0].time_offset == time_offset
    assert trained_model.models[0].dataset_name == model.models[0].dataset_name
    assert model.treatment_feature is None or trained_model.treatment_feature == model.treatment_feature
    assert model.treatment_value is None or trained_model.treatment_value == model.treatment_value
    assert model.treatment_time is None or trained_model.treatment_time == model.treatment_time
    assert (treatment_feature is None and treatment_time is None and treatment_value is None) or \
           (treatment_feature is not None and treatment_time is not None and treatment_value is not None)
    assert hidden_flag == True or (hidden_flag is False and new_model_number == 1)
    if treatment_feature is None and treatment_time is None and treatment_value is None:
        mode = 'predict'
    else:
        mode = 'treatment'
    model.set_mode(mode)

    time_list = [torch.FloatTensor(time_list).to(device) for _ in range(batch_size)]
    prediction_list, sample_ids = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_list, _, _, _, _, _, _, _, _, _, sample_id_list, _, _ = batch
            prediction = model.predict(input_list, time_list)
            prediction_list.append(prediction)
            sample_ids.append(sample_id_list)

    reorganized_prediction_list, reorganized_sample_list = [], []
    for batch_pred, batch_id in zip(prediction_list, sample_ids):
        for i in range(len(batch_id)):
            sample_result = dict()
            reorganized_sample_list.append(batch_id[i])
            for name in name_id_dict:
                name_idx = name_id_dict[name]
                feature_result = []
                for model_idx in range(len(batch_pred)):
                    feature_result.append(batch_pred[model_idx][i, :, name_idx].detach().to('cpu').numpy())
                sample_result[name] = np.array(feature_result)
            reorganized_prediction_list.append(sample_result)
    return reorganized_prediction_list, reorganized_sample_list


if __name__ == '__main__':
    main(args)