import numpy as np
from util import read_treatment_data


def main():
    data_name = 'hao_true_lmci,True,n,52,0.csv'
    hidden = True
    treat_time = 52
    middle_time = 54
    end_time = 56
    data_set = data_name.split(',')[0]

    if hidden:
        feature_list = ['a', 'tau_p', 'n', 'c']
    else:
        feature_list = ['a', 'tau_p', 'tau_o', 'n', 'c']

    data, time = read_treatment_data(data_name, feature_list)
    reorganized_data = reorganize(data)
    treatment_effect_estimation(treat_time, middle_time, end_time, time, reorganized_data)


def reorganize(data):
    reorganized_data = dict()
    sample_id_list = list()
    for sample_id in data:
        sample_id_list.append(sample_id)
        for model_name in data[sample_id]:
            if model_name not in reorganized_data:
                reorganized_data[model_name] = dict()
            if sample_id not in reorganized_data[model_name]:
                reorganized_data[model_name][sample_id] = dict()
            for feature in data[sample_id][model_name]:
                value = data[sample_id][model_name][feature]['mean']
                reorganized_data[model_name][sample_id][feature] = value

    new_data = dict()
    for model_name in reorganized_data:
        new_data[model_name] = dict()
        for sample_id in sample_id_list:
            for feature in reorganized_data[model_name][sample_id]:
                if feature not in new_data[model_name]:
                    new_data[model_name][feature] = []
                value = reorganized_data[model_name][sample_id][feature]
                new_data[model_name][feature].append(value)
    return new_data


def treatment_effect_estimation(treat_time, middle_time, end_time, time, reorganized_data):
    start_idx, middle_idx, end_idx = 1000, 1000, -1
    for i, time_point in enumerate(time):
        if time_point > treat_time:
            start_idx = i
            break
    for i, time_point in enumerate(time):
        if time_point > middle_time:
            middle_idx = i
            break
    for i, time_point in enumerate(time):
        if time_point < end_time:
            end_idx = i

    difference_dict = dict()
    for model_name in reorganized_data:
        difference_dict[model_name] = dict()
        for feature in reorganized_data[model_name]:
            predict_result = np.array(reorganized_data[model_name][feature])
            near = predict_result[:, start_idx: middle_idx+1]
            far = predict_result[:, middle_idx: end_idx]
            full = predict_result[:, start_idx: end_idx]
            difference_dict[model_name][feature] = [near, far, full]

    for model_name in difference_dict:
        near_loss_sum, far_loss_sum, full_loss_sum = 0, 0, 0
        for feature in difference_dict[model_name]:
            p_near, p_far, p_full = difference_dict[model_name][feature]
            t_near, t_far, t_full = difference_dict['oracle'][feature]
            assert len(p_near.shape) == 2
            near_loss = np.sum((t_near - p_near)**2) / (p_near.shape[0]*p_near.shape[1])
            far_loss = np.sum((t_far - p_far) ** 2) / (p_far.shape[0] * p_far.shape[1])
            full_loss = np.sum((t_full - p_full) ** 2) / (p_full.shape[0] * p_full.shape[1])
            near_loss_sum += near_loss
            far_loss_sum += far_loss
            full_loss_sum += full_loss
        near_loss_sum /= len(difference_dict[model_name])
        far_loss_sum /= len(difference_dict[model_name])
        full_loss_sum /= len(difference_dict[model_name])
        print('Model: {}, Near RMSE: {}, Far RMSE: {}, Full RMSE: {}'.format(model_name, near_loss_sum,
                                                                             far_loss_sum, full_loss_sum))


if __name__ == '__main__':
    main()
