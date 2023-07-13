import os
import pickle
import random
import numpy as np
from copy import deepcopy
from torch import FloatTensor
from torch.utils.data import Dataset, RandomSampler, DataLoader
from default_config import missing_flag_num


def format_check(dataset):
    """
    the format of data should be

    dataset: [
        # sample 1 dict
        {
            # sequential observation data of a given patient sample
            "observation": [
                # visit 1
                {
                    "visit_time": float,
                    "item_1": float,
                    "item_2": float
                    ...
                },
                # visit 2 data dict, the dict structure is strictly same as the visit 1
                {...}
                # visit 3 data dict, the dict structure is strictly same as the visit 1
                {...},
                ...
            ]
            # other dataset specific info, like sample id
            "info 1": ...,
            "info dict 2": {...},
        }
        # sample 2 dict, the dict structure is strictly same as the sample 1
        {...},
        # sample 3 dict, the dict structure is strictly same as the sample 1
        {...},
        ...
    ]

    we use -99999 to denote missing data
    """
    assert isinstance(dataset, list)
    for sample in dataset:
        # minimum seq length = 3, two for feeding model, 1 for prediction
        assert isinstance(sample['observation'], list) and len(sample['observation']) > 2
        observation_list = sample['observation']
        for i in range(len(observation_list)):
            single_visit = observation_list[i]
            # all observation has exactly the same feature
            if i != 0:
                assert len(single_visit) == len(observation_list[i - 1])
            assert 'visit_time' in single_visit
            for key in single_visit:
                if key == 'visit_time':
                    assert single_visit[key] >= 0 and isinstance(single_visit[key], float)
    return True


def id_map(single_observation):
    id_name_dict = {}
    name_id_dict = {}
    # 确保name_idx在各个dataset fraction中一致
    key_list = [key for key in single_observation]
    key_list = sorted(key_list)
    for key in key_list:
        if key == 'visit_time':
            continue
        id_name_dict[len(id_name_dict)] = key
        name_id_dict[key] = len(name_id_dict)
    return id_name_dict, name_id_dict


class SingleVisitDataset(Dataset):
    def __init__(self, dataset):
        format_check(dataset)
        self.id_name_dict, self.name_id_dict = id_map(dataset[0]['observation'][0])
        self.dataset = dataset
        self.obs_data_list, self.true_data_list = self.data_reorganize(dataset)

    def data_reorganize(self, data):
        obs_data_list, true_data_list = [], []
        for sample in data:
            for observed_data, true_data in zip(sample['observation'], sample['true_value']):
                single_obs_data, single_true_data = [0] * len(observed_data), [0] * len(true_data)
                for key in observed_data:
                    if key == 'visit_time':
                        continue
                    idx = self.name_id_dict[key]
                    single_obs_data[idx] = observed_data[key]
                obs_data_list.append(single_obs_data)
                for key in true_data:
                    if key == 'visit_time':
                        continue
                    idx = self.name_id_dict[key]
                    single_true_data[idx] = true_data[key]
                true_data_list.append(single_true_data)
        return obs_data_list, true_data_list

    def __len__(self):
        return len(self.obs_data_list)

    def __getitem__(self, index):
        return self.true_data_list[index], self.obs_data_list[index]


class SequentialVisitDataset(Dataset):
    def __init__(self, dataset):
        format_check(dataset)
        self.id_name_dict, self.name_id_dict = id_map(dataset[0]['observation'][0])
        self.dataset = dataset
        self.obs_list, self.true_list, self.time_list, self.id_list, self.init_trans_list, self.init_origin_list, \
            self.para_list = self.data_split(dataset)

    def data_split(self, data):
        obs_list, true_list, time_list, id_list, init_trans_list, init_origin_list, para_list = \
            [], [], [], [], [], [], []
        for sample in data:
            single_obs_sequence_data, single_sequence_time, single_true_sequence_data = [], [], []
            observation_sequence, true_sequence, sample_id = sample['observation'], sample['true_value'], sample['id']
            origin_init, para = sample['init'], sample['para']
            init_trans = [0] * len(self.name_id_dict)
            init_dict = {}
            assert 'visit_time' not in origin_init
            for key in origin_init:
                origin_value, trans_value = origin_init[key]['origin'], origin_init[key]['transformed']
                if key in self.name_id_dict:
                    idx = self.name_id_dict[key]
                    init_trans[idx] = trans_value
                else:
                    assert key == 'tau_o'
                init_dict[key] = origin_value

            observation_sequence = sorted(observation_sequence, key=lambda x: x['visit_time'], reverse=False)
            true_sequence = sorted(true_sequence, key=lambda x: x['visit_time'], reverse=False)
            for single_obs_visit in observation_sequence:
                single_obs_data = [0] * (len(single_obs_visit)-1)
                for key in single_obs_visit:
                    if key == 'visit_time':
                        continue
                    idx = self.name_id_dict[key]
                    single_obs_data[idx] = single_obs_visit[key]
                single_obs_sequence_data.append(single_obs_data)
                single_sequence_time.append(single_obs_visit['visit_time'])
            for single_true_visit in true_sequence:
                single_true_data = [0] * (len(single_true_visit)-1)
                for key in single_true_visit:
                    if key == 'visit_time':
                        continue
                    idx = self.name_id_dict[key]
                    single_true_data[idx] = single_true_visit[key]
                single_true_sequence_data.append(single_true_data)

            id_list.append(sample_id)
            obs_list.append(single_obs_sequence_data)
            true_list.append(single_true_sequence_data)
            time_list.append(single_sequence_time)
            init_trans_list.append(init_trans)
            init_origin_list.append(init_dict)
            para_list.append(para)
        return obs_list, true_list, time_list, id_list, init_trans_list, init_origin_list, para_list

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, index):
        return self.obs_list[index], self.true_list[index], self.time_list[index], self.id_list[index], \
            self.init_trans_list[index], self.init_origin_list[index], self.para_list[index]


class SingleVisitDataloader(DataLoader):
    def __init__(self, dataset, batch_size, sampler, mask):
        assert batch_size > 1
        self.mask = mask
        super().__init__(dataset, batch_size=batch_size, sampler=sampler,
                         collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        data = np.array(batch, np.float64)
        mask = np.array(data == self.mask, np.int64)
        return {
            'data': data,
            'mask': mask
        }


class SequentialVisitDataloader(DataLoader):
    def __init__(self, dataset, batch_size, sampler, minimum_observation, mask, reconstruct_input, predict_label,
                 device, stat_dict):
        assert batch_size > 1
        self.mask = mask
        self.reconstruct_input = reconstruct_input
        self.predict_label = predict_label
        self.minimum_observation = minimum_observation
        self.device = device
        self.stat_dict = stat_dict
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, collate_fn=self.collate_fn)

    def reorganize_batch_data(self, data):
        min_obs_time = self.minimum_observation

        # random sample label and observation
        id_list = [item[3] for item in data]
        time_list = [item[2] for item in data]
        observed_list = [item[0] for item in data]
        true_list = [item[1] for item in data]
        init_tensor_list = [item[4] for item in data]
        init_list = [item[5] for item in data]
        para_list = [item[6] for item in data]
        valid_length_list = [len(item) for item in time_list]
        # for the randint and slice character, the prediction idx need to minus one, while the observation does not
        # 此处的min_obs_time从0起数
        obs_idx_list = []
        for length in valid_length_list:
            assert min_obs_time <= length
            if length == min_obs_time:
                obs_idx_list.append(length)
            else:
                obs_idx_list.append(random.randint(min_obs_time, length - 1))

        # 复现显然是要复现obs value，预测要预测true value
        available_feature_list, available_time_list, predict_feature_list, predict_time_list = [], [], [], []
        for sample_time_list, obs_value_list, true_value_list, sample_obs_idx in \
                zip(time_list, observed_list, true_list, obs_idx_list):
            predict_time_list.append(sample_time_list[sample_obs_idx:])
            predict_feature_list.append(true_value_list[sample_obs_idx:])
            available_feature_list.append(obs_value_list[:sample_obs_idx])
            available_time_list.append(sample_time_list[:sample_obs_idx])
        return predict_time_list, predict_feature_list, available_time_list, available_feature_list, id_list, \
            init_tensor_list, init_list, para_list

    def collate_fn(self, data):
        predict_time_list, predict_feature_list, available_time_list, available_feature_list, id_list, \
            init_trans_list, init_origin_list, para_list = self.reorganize_batch_data(data)
        data = []
        for pred_time, pred_feature, avail_time, ava_feature, sample_id, init_trans, init_origin, para in \
                zip(predict_time_list, predict_feature_list, available_time_list, available_feature_list, id_list,
                    init_trans_list, init_origin_list, para_list):
            data.append([pred_time, pred_feature, avail_time, ava_feature, sample_id, init_trans, init_origin, para])

        input_feature_list, input_time_list, input_mask_list, label_feature_list, label_time_list, label_mask_list, \
            type_list, input_len_list, label_len_list, sample_id_list, input_init_trans_list, input_init_origin_list,\
            input_para_list = [], [], [], [], [], [], [], [], [], [], [], [], []
        for item in data:
            pred_time, pred_feature, avail_time, avail_feature, sample_id, init_trans, init_origin, para = item
            sample_id_list.append(sample_id)
            # 如果要在初始值用init，要额外补一个hidden dimension（当hidden 为True时）
            input_init_trans_list.append(init_trans + [1])
            input_init_origin_list.append(init_origin)
            input_para_list.append(para)

            input_feature_mask = np.array(avail_feature) == missing_flag_num
            input_feature = (1 - input_feature_mask) * np.array(avail_feature)
            input_feature_list.append(FloatTensor(input_feature))
            input_time_list.append(FloatTensor(np.array(deepcopy(avail_time))))
            input_mask_list.append(FloatTensor(input_feature_mask))

            pred_feature_mask = np.array(pred_feature) == missing_flag_num
            pred_feature = (1 - pred_feature_mask) * np.array(pred_feature)
            pred_time = deepcopy(pred_time)

            if self.reconstruct_input and (not self.predict_label):
                label_feature_list.append(FloatTensor(deepcopy(input_feature)))
                label_time_list.append(FloatTensor(deepcopy(avail_time)))
                label_mask_list.append(FloatTensor(deepcopy(input_feature_mask)))
                label_len_list.append(len(avail_time))
                type_list.append(FloatTensor([1]*len(avail_time)))
            elif (not self.reconstruct_input) and self.predict_label:
                label_feature_list.append(FloatTensor(deepcopy(pred_feature)))
                label_time_list.append(FloatTensor(deepcopy(pred_time)))
                label_mask_list.append(FloatTensor(deepcopy(pred_feature_mask)))
                label_len_list.append(len(pred_time))
                type_list.append(FloatTensor([2] * len(pred_time)))
            elif self.reconstruct_input and self.predict_label:
                if len(pred_feature) == 0:
                    label_feature = input_feature
                    label_mask = input_feature_mask
                    label_time = deepcopy(avail_time)
                else:
                    label_feature = np.concatenate([input_feature, pred_feature], axis=0)
                    label_mask = np.concatenate([input_feature_mask, pred_feature_mask], axis=0)
                    label_time = deepcopy(avail_time)
                    label_time.extend(pred_time)
                label_len_list.append(len(label_time))
                label_feature_list.append(FloatTensor(label_feature))
                label_time_list.append(FloatTensor(label_time))
                label_mask_list.append(FloatTensor(label_mask))
                type_list.append(FloatTensor([1] * len(avail_time) + [2] * len(pred_time)))
            else:
                raise ValueError('')

        concat_input = []
        for i in range(len(input_feature_list)):
            concat_sample = []
            for j in range(len(input_feature_list[i])):
                visit_feature = input_feature_list[i][j]
                visit_mask = input_mask_list[i][j]
                visit_time = input_time_list[i][j]
                visit_data = [visit_time]
                visit_data.extend(visit_feature)
                visit_data.extend(visit_mask)
                concat_sample.append(visit_data)
            input_len_list.append(len(concat_sample))
            concat_input.append(FloatTensor(concat_sample))

        concat_input = [item.to(self.device) for item in concat_input]
        input_feature_list = [item.to(self.device) for item in input_feature_list]
        input_time_list = [item.to(self.device) for item in input_time_list]
        input_mask_list = [item.to(self.device) for item in input_mask_list]
        label_feature_list = [item.to(self.device) for item in label_feature_list]
        label_time_list = [item.to(self.device) for item in label_time_list]
        label_mask_list = [item.to(self.device) for item in label_mask_list]
        type_list = [item.to(self.device) for item in type_list]
        input_init_trans_list = FloatTensor(np.array(input_init_trans_list)).to(self.device)
        return concat_input, input_feature_list, input_time_list, input_mask_list, label_feature_list,\
            label_time_list, label_mask_list, type_list, input_len_list, label_len_list, sample_id_list, \
            input_init_trans_list, input_init_origin_list, input_para_list


def main():
    batch_size = 16
    min_obs = 2
    mask_tag = -1
    reconstruct_input = True
    predict_label = True
    data_folder = os.path.abspath('../resource/simulated_data/')
    false_data_path = os.path.join(data_folder, 'sim_data_hidden_False_group_lmci_personal_0_type_random.pkl')
    true_data_path = os.path.join(data_folder, 'sim_data_hidden_True_group_lmci_personal_0_type_random.pkl')
    hidden_false_data = pickle.load(open(false_data_path, 'rb'))
    hidden_true_data = pickle.load(open(true_data_path, 'rb'))
    device = 'cpu'

    for data in [hidden_true_data['data'], hidden_false_data['data']]:
        for key in data:
            data_split = data[key]
            dataset_1 = SingleVisitDataset(data_split)
            dataset_2 = SingleVisitDataset(data_split)
            dataset_3 = SequentialVisitDataset(data_split)
            dataset_4 = SequentialVisitDataset(data_split)
            for dataset in [dataset_1, dataset_2]:
                sampler = RandomSampler(dataset)
                dataloader = SingleVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag)
                for _ in dataloader:
                    print('')
            for dataset in [dataset_3, dataset_4]:
                sampler = RandomSampler(dataset)
                dataloader = SequentialVisitDataloader(
                    dataset, batch_size, sampler=sampler, mask=mask_tag, minimum_observation=min_obs,
                    reconstruct_input=reconstruct_input, predict_label=predict_label, device=device)
                for _ in dataloader:
                    print('')


if __name__ == "__main__":
    main()
