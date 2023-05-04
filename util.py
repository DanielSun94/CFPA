from default_config import logger
from data_preprocess.data_loader import SequentialVisitDataloader, SequentialVisitDataset, RandomSampler
import pickle


def get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation, reconstruct_input,
                    predict_label):
    # to be revised in future for multi dataset
    logger.info('dataset name: {}'.format(dataset_name))
    dataset_dict = {}
    for split in 'train', 'valid', 'test':
        dataset_dict[split] = pickle.load(open(data_path, 'rb'))['data'][split]
    dataloader_dict = {}
    for split in 'train', 'valid', 'test':
        dataset = SequentialVisitDataset(dataset_dict[split])
        sampler = RandomSampler(dataset)
        dataloader = SequentialVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                               minimum_observation=minimum_observation,
                                               reconstruct_input=reconstruct_input, predict_label=predict_label)
        dataloader_dict[split] = dataloader
    return dataloader_dict
