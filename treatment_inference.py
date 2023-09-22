import csv
import os.path
import numpy as np
from default_config import ckpt_folder, args, treatment_result_inference_folder
from util import get_data_loader, OracleHaoModel, get_oracle_causal_graph, OracleAutoModel, OracleZhengModel
from torch import FloatTensor, load, no_grad, reshape, stack
from model.treatment_effect_evaluation import TreatmentEffectEstimator

def main(argument):
    assert argument['hidden_flag'] == "True" or argument['hidden_flag'] == "False"
    assert argument['reconstruct_input'] == "True" or argument['reconstruct_input'] == "False"
    assert argument['predict_label'] == "True" or argument['predict_label'] == "False"

    device = argument['device']
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
    # constraint = argument['constraint_type']

    # hao
    if dataset_name == 'hao_true_lmci':
        inference_model_name_dict = {
            # 'LODE': ('treatment.TEP.hao_true_lmci.True.20230426050205000579.0.0.model', False),
            # 'CTP': ('treatment.TEP.hao_true_lmci.True.20230426050204978059.0.0.model', False),
            # 'NODE': ('treatment.TEP.hao_true_lmci.True.20230426050204991342.0.0.model', False),
            # 'TE-CDE': ('treatment.TEP.hao_true_lmci.True.20230426063429908071.0.0.model', False),
            # 'CF-ODE': ('treatment.TEP.hao_true_lmci.True.20230426050204991414.0.0.model', False),
            # # 'NGM': ('treatment.TEP.hao_true_lmci.True.20230426050204989739.0.0.model', False),
            # 'CTP02': ('treatment.TEP.hao_true_lmci.True.20230508044453770108.25.3000.model', False, 2),
            # 'CTP04': ('treatment.TEP.hao_true_lmci.True.20230508044453736069.25.3000.model', False, 4),
            'CTP081': ('treatment.TEP.hao_true_lmci.True.20230920161102476025.42.1350.model', False, 8),
            # 'CTP082': ('treatment.TEP.hao_true_lmci.True.20230920160219722836.14.450.model', False, 8),
            # 'CTP16': ('treatment.TEP.hao_true_lmci.True.20230508044453875411.25.3000.model', False, 16),
            # 'CTP32': ('treatment.TEP.hao_true_lmci.True.20230508044454111814.23.2800.model', False, 32),
            # 'CTP64': ('treatment.TEP.hao_true_lmci.True.20230508044454093056.11.1340.model', False, 64),
        }
        use_hidden = "True"
    elif dataset_name == 'zheng':
        inference_model_name_dict = {
            # 'LODE': ('treatment.TEP.zheng.False.20230425045910181387.0.0.model', False),
            # 'CTP': ('treatment.TEP.zheng.False.20230426095624827475.0.0.model', False),
            # 'NODE': ('treatment.TEP.zheng.False.20230425045910207057.0.0.model', False),
            # 'TE-CDE': ('treatment.TEP.zheng.False.20230426113604097993.0.0.model', False),
            # 'CF-ODE': ('treatment.TEP.zheng.False.20230426113604591044.0.0.model', False),
            # 'NGM': ('treatment.TEP.zheng.False.20230425045910261224.0.0.model', False),
            # 'CTP02': ('treatment.TEP.zheng.False.20230508044454259063.37.3000.model', False, 2),
            # 'CTP04': ('treatment.TEP.zheng.False.20230508044454389246.37.3000.model', False, 4),
            'CTP081': ('treatment.TEP.hao_true_lmci.True.20230920160219722836.56.1800.model', False, 8),
            # 'CTP082': ('treatment.TEP.zheng.False.20230907171927474548.93.3000.model', False, 8),
            # 'CTP16': ('treatment.TEP.zheng.False.20230508044454583704.37.3000.model', False, 16),
            # 'CTP32': ('treatment.TEP.zheng.False.20230508044454528166.37.3000.model', False, 32),
            # 'CTP64': ('treatment.TEP.zheng.False.20230508044454820893.24.1920.model', False, 64),
        }
        use_hidden = "False"
    elif dataset_name == 'auto25':
        inference_model_name_dict = {
            # 'LODE': ('treatment.TEP.auto25.False.20230425045910068283.19.1560.model', False),
            # 'CTP': ('treatment.TEP.auto25.False.20230425045910033893.8.640.model', False),
            # 'NODE': ('treatment.TEP.auto25.False.20230425045910040216.7.620.model', False),
            # 'TE-CDE': ('treatment.TEP.auto25.False.20230425045910040216.0.0.model', False),
            # 'CF-ODE': ('treatment.TEP.auto25.False.20230425045910040216.3.300.model', False),
            # 'NGM': ('treatment.TEP.auto25.False.20230425045910055872.8.680.model', False),
            # 'CTP02': ('treatment.TEP.auto25.False.20230508044454571387.16.2000.model', False, 2),
            # 'CTP04': ('treatment.TEP.auto25.False.20230508044454922726.16.2000.model', False, 4),
            'CTP08': ('treatment.TEP.zheng.False.20230907171927474548.93.3000.model', False, 8),
            # 'CTP16': ('treatment.TEP.auto25.False.20230508044455097134.16.2000.model', False, 16),
            # 'CTP32': ('treatment.TEP.auto25.False.20230508044454956351.16.2000.model', False, 32),
            # 'CTP64': ('treatment.TEP.auto25.False.20230508044455132028.7.900.model', False, 64),
        }
        use_hidden = "False"
    elif dataset_name == 'auto50':
        inference_model_name_dict = {
            # 'LODE': ('treatment.TEP.auto50.False.20230425045910112885.11.900.model', False),
            # 'CTP': ('treatment.TEP.auto50.False.20230425045910229400.3.280.model', False),
            # 'NODE': ('treatment.TEP.auto50.False.20230425045910042825.3.240.model', False),
            # 'TE-CDE': ('treatment.TEP.auto50.False.20230425045910042825.1.120.model', False),
            # 'CF-ODE': ('treatment.TEP.auto50.False.20230425045910042825.0.0.model', False),
            # 'NGM': ('treatment.TEP.auto50.False.20230425045910078953.3.280.model', False),
            'CTP02': ('treatment.TEP.auto50.False.20230508044455151617.12.1000.model', False, 2),
            'CTP04': ('treatment.TEP.auto50.False.20230508044455391233.12.1000.model', False, 4),
            'CTP08': ('treatment.TEP.auto50.False.20230508044455362061.12.1000.model', False, 8),
            'CTP16': ('treatment.TEP.auto50.False.20230508044455442166.12.1000.model', False, 16),
            'CTP32': ('treatment.TEP.auto50.False.20230508044455393160.9.720.model', False, 32),
            'CTP64': ('treatment.TEP.auto50.False.20230508044455762910.5.420.model', False, 64),
        }
        use_hidden = "False"
    else:
        raise ValueError('')


    dataloader_dict, name_id_dict, oracle_graph, id_type_list, stat_dict = \
        get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                        reconstruct_input, predict_label, device=device)
    test_dataset = dataloader_dict['test']
    t_idx = name_id_dict[t_feature]

    # 干预要经过正态转换
    mean, std = stat_dict[t_feature]
    origin_t_value = t_value
    t_value = (t_value - mean) / std
    time_list = np.array([(i + 1) * 0.05 * (obs_time - time_offset) + time_offset for i in range(20)])

    oracle_treatment_dataset = generate_oracle_behavior(
        use_hidden, test_dataset, dataset_name, stat_dict, t_feature, t_time, time_list, origin_t_value, time_offset,
        oracle_graph)
    oracle_original_dataset = generate_oracle_behavior(
        use_hidden, test_dataset, dataset_name, stat_dict, None, None, time_list, None, time_offset,
        oracle_graph)

    data_dict = dict()
    for model_name in inference_model_name_dict:
        model_file, true_causal, new_model_number = inference_model_name_dict[model_name]

        if true_causal:
            oracle_graph = get_oracle_causal_graph(name_id_dict, use_hidden, 'use_data', oracle_graph)
        else:
            oracle_graph = get_oracle_causal_graph(name_id_dict, use_hidden, 'not_causal', oracle_graph)

        model_treatment_dataset = generate_model_behavior(
            use_hidden, test_dataset, dataset_name, t_feature, t_time, time_list, t_value, t_idx, new_model_number,
            model_name, model_file, oracle_graph, id_type_list, name_id_dict, device, argument
        )
        # model_origin_dataset = generate_model_behavior(
        #     use_hidden, test_dataset, dataset_name, None, None, time_list, None, None, model_name,
        #     model_file, oracle_graph, id_type_list, name_id_dict, device, argument
        # )
        data_dict[model_name] = model_treatment_dataset
        # data_dict[model_name + '_origin'] = model_origin_dataset
    data_dict['oracle'] = oracle_treatment_dataset
    data_dict['oracle_origin'] = oracle_original_dataset

    fused_dict = fuse_result(data_dict, time_list)
    file_name = "{},{},{},{},{}.csv"\
        .format(dataset_name, use_hidden, t_feature, t_time, origin_t_value)
    save_result(fused_dict, file_name)

def save_result(fused_dict, file_name):
    result_dict, time_list = fused_dict
    data = []
    head = ['sample_id', 'model', 'feature', 'data_type']
    for item in time_list:
        head.append(item)
    data.append(head)
    for sample_id in result_dict:
        for model in result_dict[sample_id]:
            for feature in result_dict[sample_id][model]:
                result = result_dict[sample_id][model][feature]
                if len(result.shape) > 1:
                    result = result.reshape([-1, len(time_list)])
                    mean = np.mean(result, axis=0).tolist()
                    max_ = np.max(result, axis=0).tolist()
                    min_ = np.min(result, axis=0).tolist()
                    data.append([sample_id, model, feature, 'mean'] + mean)
                    data.append([sample_id, model, feature, 'max'] + max_)
                    data.append([sample_id, model, feature, 'min'] + min_)
                else:
                    data.append([sample_id, model, feature, 'mean'] + [item for item in result])
                    data.append([sample_id, model, feature, 'max'] + [0] * len(result))
                    data.append([sample_id, model, feature, 'min'] + [0] * len(result))

    with open(os.path.join(treatment_result_inference_folder, file_name), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data)



def fuse_result(data_dict, time_list):
    result_dict = dict()
    for i in range(len(data_dict['oracle'][1])):
        assert data_dict['oracle'][1][i] not in result_dict
        result_dict[data_dict['oracle'][1][i]] = {'oracle': data_dict['oracle'][0][i]}
    for i in range(len(data_dict['oracle'][1])):
        for key in data_dict:
            if key == 'oracle':
                continue
            result_dict[data_dict[key][1][i]][key] = data_dict[key][0][i]
            result_dict[data_dict[key][1][i]][key] = data_dict[key][0][i]
            result_dict[data_dict[key][1][i]][key] = data_dict[key][0][i]
    return result_dict, time_list


def get_oracle_model(use_hidden, model_name, para_dict, init_dict, time_offset, stat_dict):
    if 'hao' in model_name:
        return OracleHaoModel(use_hidden, para_dict, init_dict, time_offset, stat_dict)
    elif 'zheng' in model_name:
        return OracleZhengModel(use_hidden, para_dict, init_dict, time_offset, stat_dict)
    elif 'auto50' in model_name:
        return OracleAutoModel(use_hidden, init_dict, time_offset, stat_dict, 50)
    elif 'auto25' in model_name:
        return OracleAutoModel(use_hidden, init_dict, time_offset, stat_dict, 25)
    else:
        raise ValueError('')


def generate_oracle_behavior(hidden_flag, dataloader, dataset, stat_dict, treatment_feature, treatment_time,
                             time_list, treatment_value, time_offset, oracle_graph):
    prediction_list, sample_ids = [], []
    for batch in dataloader:
        _, _, _, _, _, _, _, _, _, _, sample_id_list, _, init_list, para_list = batch
        for sample_id, init, para in zip(sample_id_list, init_list, para_list):
            model = get_oracle_model(hidden_flag, dataset, para, init, time_offset, stat_dict)
            model.set_treatment(treatment_feature, treatment_time, treatment_value)
            prediction = model.inference(time_list)
            prediction_list.append(prediction)
            sample_ids.append(sample_id)
    return prediction_list, sample_ids


def generate_model_behavior(hidden_flag, dataloader, dataset_name, treatment_feature, treatment_time, time_list,
                            treatment_value, treatment_idx, new_model_number, model_name, inference_model_file,
                            oracle_graph, id_type_list, name_id_dict, device, argument):
    if 'Linear' in model_name or 'linear' in model_name or 'LODE' in model_name:
        non_linear = 'False'
    else:
        non_linear = "True"
    # new_model_number = argument['treatment_new_model_number']
    batch_size = dataloader.batch_size
    process_name = argument['process_name']
    optimize_method = argument['treatment_optimize_method']
    sample_multiplier = argument['treatment_sample_multiplier']
    model_args = {
        'init_model_name': None,
        'hidden_flag': hidden_flag,
        'input_size': argument['input_size'],
        'distribution_mode': argument['distribution_mode'],
        'batch_first': argument['batch_first'],
        'hidden_size': argument['hidden_size'],
        'bidirectional': argument['init_net_bidirectional'],
        'device': argument['device'],
        'dataset_name': argument['dataset_name'],
        'time_offset': argument['time_offset'],
        'non_linear_mode': non_linear
    }

    model = TreatmentEffectEstimator(
        dataset_name=dataset_name, device=device, treatment_idx=treatment_idx, oracle_graph=oracle_graph,
        batch_size=batch_size, treatment_feature=treatment_feature, new_model_number=new_model_number,
        id_type_list=id_type_list, model_args=model_args, treatment_time=treatment_time,
        process_name=process_name, treatment_value=treatment_value, optimize_method=optimize_method
    )
    trained_model = load(os.path.join(ckpt_folder, inference_model_file))
    model.load_state_dict(trained_model)
    model.to(device)

    assert argument['hidden_flag'] == 'True' or argument['hidden_flag'] == "False"
    # assert True if argument['hidden_flag'] == "True" else False == hidden_flag
    # 注释掉是因为保存的是state dict，以下信息均无效了
    # assert trained_model.new_model_number == model.new_model_number
    # assert trained_model.models[0].hidden_flag == model.models[0].hidden_flag
    # assert trained_model.models[0].time_offset == time_offset
    # assert trained_model.models[0].dataset_name == model.models[0].dataset_name
    # assert model.treatment_feature is None or trained_model.treatment_feature == model.treatment_feature
    # assert model.treatment_value is None or trained_model.treatment_value == model.treatment_value
    # assert model.treatment_time is None or trained_model.treatment_time == model.treatment_time
    # assert (treatment_feature is None and treatment_time is None and treatment_value is None) or \
    #        (treatment_feature is not None and treatment_time is not None and treatment_value is not None)
    # assert hidden_flag == True or (hidden_flag is False and new_model_number == 1)
    if treatment_feature is None and treatment_time is None and treatment_value is None:
        mode = 'predict'
    else:
        mode = 'treatment'
    model.set_mode(mode)

    time_list = [FloatTensor(time_list).to(device) for _ in range(batch_size)]
    prediction_list, sample_ids = [], []
    model.set_sample_multiplier(sample_multiplier)
    model_num = len(model.models)
    with no_grad():
        for batch in dataloader:
            input_list, sample_id_list = batch[0], batch[10]
            prediction = model.predict(input_list, time_list)
            prediction = stack([stack(item) for item in prediction], dim=0)
            prediction_list.append(reshape(prediction, (model_num, sample_multiplier, -1,
                                                        prediction.shape[2], prediction.shape[3])))
            sample_ids.append(sample_id_list)

    reorganized_prediction_list, reorganized_sample_list = [], []
    for batch_pred, batch_id in zip(prediction_list, sample_ids):
        for i in range(len(batch_id)):
            sample_result = dict()
            reorganized_sample_list.append(batch_id[i])

            for name in name_id_dict:
                name_idx = name_id_dict[name]
                feature_result = []
                for model_idx in range(model_num):
                    feature_result.append(batch_pred[model_idx, :, i, :, name_idx].detach().to('cpu').numpy())
                sample_result[name] = np.array(feature_result)
            reorganized_prediction_list.append(sample_result)
    return reorganized_prediction_list, reorganized_sample_list


if __name__ == '__main__':
    main(args)