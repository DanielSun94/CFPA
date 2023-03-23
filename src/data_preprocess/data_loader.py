import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, RandomSampler, DataLoader


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

    we use -1 to denote missing data
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
                    assert single_visit[key] > 0 and isinstance(single_visit[key], float)
                else:
                    if single_visit[key] < 0:
                        assert single_visit[key] == -1
                    else:
                        assert isinstance(single_visit[key], float)
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
        self.single_sample_list = self.data_reorganize(dataset)

    def data_reorganize(self, data):
        single_data_list = []
        for sample in data:
            for single_visit in sample['observation']:
                single_data = [0 for _ in range(len(single_visit) - 1)]
                for key in single_visit:
                    if key == 'visit_time':
                        continue
                    idx = self.name_id_dict[key]
                    single_data[idx] = single_visit[key]
                single_data_list.append(single_data)
        return single_data_list

    def __len__(self):
        return len(self.single_sample_list)

    def __getitem__(self, index):
        return self.single_sample_list[index]

    @staticmethod
    def collate(batch):
        print('')


class SequentialVisitDataset(Dataset):
    def __init__(self, dataset):
        format_check(dataset)
        self.id_name_dict, self.name_id_dict = id_map(dataset[0]['observation'][0])
        self.dataset = dataset
        self.data_list, self.time_list = self.data_split(dataset)

    def data_split(self, data):
        data_list = []
        time_list = []
        for sample in data:
            single_sequence_data, single_sequence_time = [], []
            sequence = sample['observation']
            sequence = sorted(sequence, key=lambda x: x['visit_time'], reverse=False)
            for single_visit in sequence:
                single_sequence_time.append(single_visit['visit_time'])
                single_data = [0 for _ in range(len(single_visit) - 1)]
                for key in single_visit:
                    if key == 'visit_time':
                        continue
                    idx = self.name_id_dict[key]
                    single_data[idx] = single_visit[key]
                single_sequence_data.append(single_data)
            data_list.append(single_sequence_data)
            time_list.append(single_sequence_time)
        return data_list, time_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index], self.time_list[index]


class SequentialDataloader(DataLoader):
    def __init__(self, dataset, batch_size, sampler, minimum_observation, mask, only_predict_next=False):
        assert batch_size > 1
        self.real_batch_size = batch_size
        self.mask = mask
        self.only_predict_next = only_predict_next
        self.minimum_observation = minimum_observation
        super().__init__(dataset, batch_size=len(dataset), sampler=sampler, collate_fn=self.collate_fn)

    def reorganize_batch_data(self, data):
        min_obser_time = self.minimum_observation
        batch_size = self.real_batch_size
        sample_id_list = np.random.randint(0, len(data), [batch_size])

        batch_data = []
        for idx in sample_id_list:
            batch_data.append(data[idx])
        # random sample label and observation
        time_list = [item[1] for item in batch_data]
        feature_list = [item[0] for item in batch_data]
        valid_length_list = [len(item) for item in time_list]
        # for the randint and slice character, the prediction idx need to minus one, while the observation does not
        prediction_idx_list = [random.randint(min_obser_time, length - 1) for length in valid_length_list]
        # 此处这么设计是为了能够让observation预测较远的结果，如果永远是预测下一次的话，模型很难做远期预测
        if self.only_predict_next:
            observation_idx_list = [pred_idx for pred_idx in prediction_idx_list]
        else:
            observation_idx_list = [random.randint(min_obser_time, pred_idx) for pred_idx in prediction_idx_list]

        available_feature_list, available_time_list, predict_feature_list, predict_time_list = [], [], [], []
        for sample_time_list, sample_feature_list, sample_obser_idx, sample_pred_idx in \
                zip(time_list, feature_list, observation_idx_list, prediction_idx_list):
            assert sample_obser_idx <= sample_pred_idx
            predict_time_list.append(sample_time_list[sample_pred_idx])
            predict_feature_list.append(sample_feature_list[sample_pred_idx])
            available_feature_list.append(sample_feature_list[:sample_obser_idx])
            available_time_list.append(sample_time_list[:sample_obser_idx])
        return predict_time_list, predict_feature_list, available_time_list, available_feature_list

    def vectorize_nested_list(self, nested_list):
        mask_num = self.mask
        if isinstance(nested_list[0], int) or isinstance(nested_list[0], float):
            feature = np.array(nested_list, dtype=np.float64)
            mask = np.array(feature == mask_num, dtype=np.int64)
            length = np.ones_like(nested_list, dtype=np.int64)
            return feature, mask, length

        assert isinstance(nested_list[0][0], list)
        length = np.array([len(item) for item in nested_list], dtype=np.int64)
        max_length = np.max(length)

        new_nested_list = []
        sample_dim = len(nested_list[0][0])
        for sample_seq in nested_list:
            new_sample_seq = []
            for single_sample in sample_seq:
                new_sample_seq.append(single_sample)
            for i in range(max_length-len(sample_seq)):
                new_sample_seq.append([-1 for _ in range(sample_dim)])
            new_nested_list.append(new_sample_seq)

        new_nested_data = np.array(new_nested_list, dtype=np.float64)
        mask = np.array(new_nested_data == -1, dtype=np.int64)
        new_nested_data = new_nested_data + mask
        return new_nested_data, mask, length

    def collate_fn(self, data):
        predict_time_list, predict_feature_list, available_time_list, available_feature_list = \
            self.reorganize_batch_data(data)
        pred_feature_data, pred_feature_mask, pred_feature_len = self.vectorize_nested_list(predict_feature_list)
        pred_time_data, _, __ = self.vectorize_nested_list(predict_time_list)
        avai_feature_data, avai_feature_mask, avai_feature_len = self.vectorize_nested_list(available_feature_list)
        avai_time_data, avai_time_mask, avai_time_len = self.vectorize_nested_list(available_time_list)
        return [avai_feature_data, avai_feature_mask, avai_feature_len], \
               [avai_time_data, avai_time_mask, avai_time_len],\
               [pred_feature_data, pred_feature_mask, pred_feature_len], pred_time_data


def main():
    batch_size = 16
    min_obser = 2
    mask_tag = 0
    data_folder = os.path.abspath('../../resource/simulated_data/')
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
            for dataset in [dataset_3, dataset_4]:
                sampler = RandomSampler(dataset)
                dataloader = SequentialDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                                  minimum_observation=min_obser)
                for _ in dataloader:
                    pass
            for dataset in [dataset_1, dataset_2]:
                sampler = RandomSampler(dataset)
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=dataset.collate)
                for _ in dataloader:
                    pass


if __name__ == "__main__":
    main()
