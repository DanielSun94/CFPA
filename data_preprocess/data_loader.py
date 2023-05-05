import os
import pickle
import random
import numpy as np
from copy import deepcopy
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
    def __init__(self, dataset, batch_size, sampler, minimum_observation, mask, reconstruct_input, predict_label):
        assert batch_size > 1
        self.real_batch_size = batch_size
        self.mask = mask
        self.reconstruct_input = reconstruct_input
        self.predict_label = predict_label
        self.minimum_observation = minimum_observation
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
        obs_idx_list = [random.randint(min_obs_time, length - 1) for length in valid_length_list]

        available_feature_list, available_time_list, predict_feature_list, predict_time_list = [], [], [], []
        for sample_time_list, obs_value_list, true_value_list, sample_obs_idx in \
                zip(time_list, observed_list, true_list, obs_idx_list):
            predict_time_list.append(sample_time_list[sample_obs_idx:])
            predict_feature_list.append(true_value_list[sample_obs_idx:])
            available_feature_list.append(obs_value_list[:sample_obs_idx])
            available_time_list.append(sample_time_list[:sample_obs_idx])
        return predict_time_list, predict_feature_list, available_time_list, available_feature_list

    def collate_fn(self, data):
        def feed_list(input_feature, input_time, output_feature, output_time, type_idx, input_features, input_times,
                      input_masks, label_features, label_times, label_masks, types):
            input_time_copy = deepcopy(input_time)
            input_feature_copy = deepcopy(input_feature)
            input_feature_mask = np.array(input_feature_copy) == missing_flag_num
            input_feature_copy = (1 - input_feature_mask) * input_feature_copy
            output_time_copy = deepcopy(output_time)
            output_feature_copy = deepcopy(output_feature)
            output_feature_mask = np.array(output_feature_copy) == missing_flag_num
            output_feature_copy = (1 - output_feature_mask) * output_feature_copy
            input_features.append(input_feature_copy.tolist())
            input_times.append(input_time_copy)
            input_masks.append(input_feature_mask.tolist())
            label_features.append(output_feature_copy.tolist())
            label_times.append(output_time_copy)
            label_masks.append(output_feature_mask.tolist())
            types.append(type_idx)

        predict_time_list, predict_feature_list, available_time_list, available_feature_list = \
            self.reorganize_batch_data(data)
        data = []
        for pred_time, pred_feature, avail_time, ava_feature in \
                zip(predict_time_list, predict_feature_list, available_time_list, available_feature_list):
            data.append([pred_time, pred_feature, avail_time, ava_feature])
        data = sorted(data, key=lambda x: len(x[3]), reverse=True)

        input_feature_list, input_time_list, input_mask_list, label_feature_list, label_time_list, label_mask_list, \
            type_list = [], [], [], [], [], [], []
        for item in data:
            pred_time, pred_feature, avail_time, avail_feature = item
            if self.reconstruct_input:
                for time, feature in zip(avail_time, avail_feature):
                    feed_list(avail_feature, avail_time, feature, time, 1, input_feature_list, input_time_list,
                              input_mask_list, label_feature_list, label_time_list, label_mask_list, type_list)
            if self.predict_label:
                for time, feature in zip(pred_time, pred_feature):
                    feed_list(avail_feature, avail_time, feature, time, 0, input_feature_list, input_time_list,
                              input_mask_list, label_feature_list, label_time_list, label_mask_list, type_list)

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
            concat_input.append(concat_sample)
        return concat_input, input_feature_list, input_time_list, input_mask_list, label_feature_list,\
            label_time_list, label_mask_list, type_list


    # def vectorize_nested_list(self, nested_list):
    #     mask_num = self.mask
    #     # pred time
    #     if isinstance(nested_list[0], int) or isinstance(nested_list[0], float):
    #         feature = np.array(nested_list, dtype=np.float64)
    #         mask = np.array(feature == mask_num, dtype=np.int64)
    #         length = np.ones_like(nested_list, dtype=np.int64)
    #         return feature, mask, length
    #
    #     # available time, pred feature
    #     if isinstance(nested_list[0][0], int) or isinstance(nested_list[0][0], float):
    #         new_nested_list = []
    #         length = np.array([len(item) for item in nested_list], dtype=np.int64)
    #         max_length = np.max(length)
    #         for sample in nested_list:
    #             new_sample = []
    #             for item in sample:
    #                 new_sample.append(item)
    #             for i in range(max_length - len(new_sample)):
    #                 new_sample.append(-1)
    #             new_nested_list.append(new_sample)
    #         feature = np.array(new_nested_list, dtype=np.float64)
    #         mask = np.array(feature == mask_num, dtype=np.int64)
    #         return feature, mask, length
    #
    #     assert isinstance(nested_list[0][0], list) and \
    #            (isinstance(nested_list[0][0][0], float) or isinstance(nested_list[0][0][0], int))
    #     length = np.array([len(item) for item in nested_list], dtype=np.int64)
    #     max_length = np.max(length)
    #
    #     new_nested_list = []
    #     sample_dim = len(nested_list[0][0])
    #     for sample_seq in nested_list:
    #         new_sample_seq = []
    #         for single_sample in sample_seq:
    #             new_sample_seq.append(single_sample)
    #         for i in range(max_length-len(sample_seq)):
    #             new_sample_seq.append([-1 for _ in range(sample_dim)])
    #         new_nested_list.append(new_sample_seq)
    #
    #     new_nested_data = np.array(new_nested_list, dtype=np.float64)
    #     mask = np.array(new_nested_data == mask_num, dtype=np.int64)
    #     return new_nested_data, mask, length
    #
    # def collate_fn(self, data):
    #     predict_time_list, predict_feature_list, available_time_list, available_feature_list = \
    #         self.reorganize_batch_data(data)
    #     pred_feature_data, pred_feature_mask, pred_feature_len = self.vectorize_nested_list(predict_feature_list)
    #     pred_time_data, pred_time_mask, pred_time_len = self.vectorize_nested_list(predict_time_list)
    #     avai_feature_data, avai_feature_mask, avai_feature_len = self.vectorize_nested_list(available_feature_list)
    #     avai_time_data, avai_time_mask, avai_time_len = self.vectorize_nested_list(available_time_list)
    #     return {
    #         'input': {
    #             'feature': {
    #                 'data': avai_feature_data,
    #                 'mask': avai_feature_mask,
    #                 'length': avai_feature_len
    #             },
    #             'time': {
    #                 'data': avai_time_data,
    #                 'mask': avai_time_mask,
    #                 'length': avai_time_len
    #             }
    #         },
    #         'target': {
    #             'time': {
    #                 'data': pred_time_data,
    #                 'mask': pred_time_mask,
    #                 'length': pred_time_len
    #             },
    #             'feature': {
    #                 'data': pred_feature_data,
    #                 'mask': pred_feature_mask,
    #                 'length': pred_feature_len
    #             }
    #         }
    #     }


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
                    reconstruct_input=reconstruct_input, predict_label=predict_label)
                for _ in dataloader:
                    print('')


if __name__ == "__main__":
    main()
