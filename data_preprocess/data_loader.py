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
    for key in single_observation:
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
        self.obs_list, self.true_list, self.time_list = self.data_split(dataset)

    def data_split(self, data):
        obs_list, true_list, time_list = [], [], []
        for sample in data:
            single_obs_sequence_data, single_sequence_time, single_true_sequence_data = [], [], []
            observation_sequence, true_sequence = sample['observation'], sample['true_value']
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

            obs_list.append(single_obs_sequence_data)
            true_list.append(single_true_sequence_data)
            time_list.append(single_sequence_time)
        return obs_list, true_list, time_list

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, index):
        return self.obs_list[index], self.true_list[index], self.time_list[index]


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
                 device):
        assert batch_size > 1
        self.real_batch_size = batch_size
        self.mask = mask
        self.reconstruct_input = reconstruct_input
        self.predict_label = predict_label
        self.minimum_observation = minimum_observation
        self.device = device
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, collate_fn=self.collate_fn)

    def reorganize_batch_data(self, data):
        min_obs_time = self.minimum_observation
        batch_size = self.real_batch_size
        sample_id_list = np.random.randint(0, len(data), [batch_size])

        batch_data = []
        for idx in sample_id_list:
            batch_data.append(data[idx])
        # random sample label and observation
        time_list = [item[2] for item in batch_data]
        observed_list = [item[0] for item in batch_data]
        true_list = [item[1] for item in batch_data]
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

        available_feature_list, available_time_list, predict_feature_list, predict_time_list = [], [], [], []
        for sample_time_list, obs_value_list, true_value_list, sample_obs_idx in \
                zip(time_list, observed_list, true_list, obs_idx_list):
            predict_time_list.append(sample_time_list[sample_obs_idx:])
            predict_feature_list.append(true_value_list[sample_obs_idx:])
            available_feature_list.append(obs_value_list[:sample_obs_idx])
            available_time_list.append(sample_time_list[:sample_obs_idx])
        return predict_time_list, predict_feature_list, available_time_list, available_feature_list

    def collate_fn(self, data):
        predict_time_list, predict_feature_list, available_time_list, available_feature_list = \
            self.reorganize_batch_data(data)
        data = []
        for pred_time, pred_feature, avail_time, ava_feature in \
                zip(predict_time_list, predict_feature_list, available_time_list, available_feature_list):
            data.append([pred_time, pred_feature, avail_time, ava_feature])

        input_feature_list, input_time_list, input_mask_list, label_feature_list, label_time_list, label_mask_list, \
            type_list, input_len_list, label_len_list = [], [], [], [], [], [], [], [], []
        for item in data:
            pred_time, pred_feature, avail_time, avail_feature = item
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
        return concat_input, input_feature_list, input_time_list, input_mask_list, label_feature_list,\
            label_time_list, label_mask_list, type_list, input_len_list, label_len_list


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
