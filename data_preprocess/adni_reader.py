import os
import csv
import copy
import random
from math import ceil
from default_config import data_folder
import numpy as np
import pickle


def main():
    file_path = os.path.join(data_folder, 'ADNI_merge.csv')
    save_path = os.path.join(data_folder, 'ADNI_merge_preprocessed.pkl')
    miss_placeholder = -1
    new_missing_placeholder = -99999
    data_dict, name_id_dict = read_data(file_path)
    data_type_dict = data_type_identify(data_dict, name_id_dict, miss_placeholder)
    reorganized_data = data_reorganize(data_dict, name_id_dict)
    train_data, val_data, test_data = random_split(reorganized_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    new_train, new_valid, new_test, true_stat_dict = \
        post_preprocess(train_data, val_data, test_data, data_type_dict, miss_placeholder, new_missing_placeholder)

    pickle.dump(
        {
            'config': {'none': 'none'},
            'setting': {'none': 'none'},
            'data': {
                'train': new_train,
                'valid': new_valid,
                'test': new_test,
            },
            'stat_dict': true_stat_dict,
            'oracle_graph': {'none': 'none'},
            'feature_type_list': data_type_dict
        },
        open(save_path, 'wb')
    )


def post_preprocess(train, valid, test, data_type_dict, miss_placeholder, new_missing_placeholder, transformer=True):
    true_dict = {}
    for key in data_type_dict:
        true_dict[key] = []
    if transformer:
        for data_fraction in train, valid, test:
            for sample in data_fraction:
                true_value = sample['true_value']
                for visit in true_value:
                    for key in visit:
                        if key == 'visit_time':
                            continue
                        true_dict[key].append(visit[key])
        true_stat_dict = dict()
        for key in true_dict:
            true_stat_dict[key] = np.mean(true_dict[key]), np.std(true_dict[key])
    else:
        true_stat_dict = {key: [0, 1] for key in true_dict}

    new_train, new_valid, new_test = [], [], []
    for origin, new in zip([train, valid, test], [new_train, new_valid, new_test]):
        for sample in origin:
            new_sample = {'para': sample['para'], 'init': sample['init'], 'id': sample['id']}
            obs, true = [], []
            for origin_visit_list, new_visit_list in \
                    zip([sample['observation'], sample['true_value']], [obs, true]):
                for single_visit in origin_visit_list:
                    new_single_visit = {}
                    for key in single_visit:
                        value = single_visit[key]
                        # 如果是use hidden，默认tau_o应该是在true和obs中均不可见的
                        if value == miss_placeholder:
                            new_single_visit[key] = new_missing_placeholder
                        elif key == 'visit_time':
                            new_single_visit[key] = value / 20
                        else:
                            if data_type_dict[key] == 'c':
                                new_single_visit[key] = (value - true_stat_dict[key][0]) / true_stat_dict[key][1]
                            else:
                                new_single_visit[key] = value
                    new_visit_list.append(new_single_visit)
            new_sample['observation'] = obs
            new_sample['true_value'] = true
            new.append(new_sample)
    return new_train, new_valid, new_test, true_stat_dict


def random_split(reorganized_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1
    size = len(reorganized_data)
    idx_list = [i for i in range(len(reorganized_data))]
    random.shuffle(idx_list)
    train_data, val_data, test_data = [], [], []
    for i in range(ceil(size * train_ratio)):
        train_data.append(reorganized_data[idx_list[i]])
    for i in range(ceil(size * train_ratio), ceil(size*(train_ratio+val_ratio))):
        val_data.append(reorganized_data[idx_list[i]])
    for i in range(ceil(size*(train_ratio+val_ratio)), size):
        test_data.append(reorganized_data[idx_list[i]])
    return train_data, val_data, test_data


def data_reorganize(data_dict, name_id_dict):
    reorganized_dataset = list()
    for p_id in data_dict:
        sample_data = {'id': str(p_id)}
        results = data_dict[p_id]
        reorganized_data = []
        for visit, visit_time in results:
            visit_dict = {'visit_time': visit_time}
            for feature in name_id_dict:
                feature_idx = name_id_dict[feature]
                feature_value = visit[feature_idx]
                visit_dict[feature] = feature_value
            reorganized_data.append(visit_dict)
        sample_data['true_value'] = reorganized_data
        sample_data['observation'] = copy.deepcopy(reorganized_data)
        sample_data['init'] = {key: {'origin': 0.0, 'transformed': 0.0} for key in name_id_dict}
        sample_data['para'] = {key: 0.0 for key in name_id_dict}
        reorganized_dataset.append(sample_data)
    return reorganized_dataset

def data_type_identify(data, name_id_dict, miss_placeholder):
    data_type_list = ['d'] * len(name_id_dict)
    for p_id in data:
        result = [visit[0] for visit in data[p_id]]
        for visit in result:
            for i, feature in enumerate(visit):
                if feature != 0 and feature != 1 and feature != miss_placeholder:
                    data_type_list[i] = 'c'
    data_type_dict = dict()
    for key in name_id_dict:
        data_type_dict[key] = data_type_list[name_id_dict[key]]
    return data_type_dict


def read_data(file_path):
    data_dict = dict()
    name_id_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            if i == 0:
                for j in range(3, len(line)):
                    name_id_dict[line[j]] = len(name_id_dict)
                name_id_dict['diagnosis'] = len(name_id_dict)
                continue
            p_id, diagnosis, visit_time, features = line[0], float(line[1]), float(line[2]), line[3:]
            features = [float(item) for item in features]
            if p_id not in data_dict:
                data_dict[p_id] = []
            data_dict[p_id].append([features + [diagnosis], visit_time])
    for p_id in data_dict:
        result_list = data_dict[p_id]
        result_list = sorted(result_list, key=lambda x: x[1])
        data_dict[p_id] = result_list
    return data_dict, name_id_dict




if __name__ == '__main__':
    main()
