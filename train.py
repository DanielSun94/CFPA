from default_config import args, logger
import pickle
from data_preprocess.data_loader import SequentialVisitDataloader, SequentialVisitDataset, RandomSampler
from model.causal_trajectory_prediction import CausalTrajectoryPrediction
from torch.optim import Adam


def unit_test(argument):
    data_path = argument['data_path']
    batch_first = True if argument['batch_first'] == 'True' else False
    batch_size = argument['batch_size']
    mediate_size = argument['mediate_size']
    minimum_observation = argument['minimum_observation']
    input_size = argument['input_size']
    mask_tag = argument['mask_tag']
    hidden_size = argument['hidden_size']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    graph_type = argument['graph_type']
    constraint = argument['constraint_type']

    init_lambda = argument['init_lambda']
    init_mu = argument['init_mu']
    eta = argument['eta']
    gamma = argument['gamma']
    stop_threshold = argument['stop_threshold']

    if graph_type == 'DAG':
        data = pickle.load(open(data_path, 'rb'))['data']['train']
    elif graph_type == 'ADMG':
        data = pickle.load(open(data_path, 'rb'))['data']['train']
    else:
        raise ValueError('')

    model = CausalTrajectoryPrediction(graph_type=graph_type, constraint=constraint, input_size=input_size,
                                       hidden_size=hidden_size, batch_first=batch_first, mediate_size=mediate_size)
    optimizer = Adam(model.parameters())
    dataset = SequentialVisitDataset(data)
    sampler = RandomSampler(dataset)
    dataloader = SequentialVisitDataloader(dataset, batch_size, sampler=sampler, mask=mask_tag,
                                           minimum_observation=minimum_observation,
                                           reconstruct_input=reconstruct_input, predict_label=predict_label)

    for batch in dataloader:
        predict, label_mask, label_feature, loss = model(batch)
        constraint = model.constraint()
        final_loss = augmented_loss(loss, constraint, init_lambda, init_mu)
        final_loss.backward()
        optimizer.step()
        print('aa')


def augmented_loss(predict_loss, constraint, lambda_coefficient, mu_coefficient):
    loss = predict_loss - lambda_coefficient * constraint - mu_coefficient/2 * constraint**2
    return loss


def train(argument):
    pass


