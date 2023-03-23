import os
import torch
import pickle


class CausalDiscovery(torch.nn.Module):
    def __init__(self, graph_type, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert graph_type == 'DAG' or graph_type == 'ADMG'
        assert (graph_type == 'DAG' and constraint == 'default') or (constraint in {'ancestral', 'bow_free', 'arid'})

        self.__graph_type = graph_type
        self.__constraint = constraint
        if graph_type == 'DAG':
            self.model = DAG(constraint)
        elif graph_type == 'ADMG':
            self.model = ADMG(constraint)

    def forward(self):
        pass


class DAG(torch.nn.Module):
    def __init__(self, constraint, *args, **kwargs):
        self.constraint = constraint
        super().__init__(*args, **kwargs)


class ADMG(torch.nn.Module):
    def __init__(self, constraint, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint = constraint


def unit_test():
    data_folder = os.path.abspath('../../resource/simulated_data/')
    hidden_false_data = os.path.join(data_folder, 'sim_data_hidden_False_group_lmci_personal_0_type_random.pkl')
    hidden_true_data = os.path.join(data_folder, 'sim_data_hidden_True_group_lmci_personal_0_type_random.pkl')

    test_seq = [
        ['DAG', 'default'],
        ['ADMG', 'ancestral'],
        ['ADMG', 'bow_free'],
        ['ADMG', 'arid'],
    ]
    for (graph_type, constraint) in test_seq:
        model = CausalDiscovery(graph_type=graph_type, constraint=constraint)
        if graph_type == 'DAG':
            data = pickle.load(open(hidden_false_data, 'rb'))
        elif graph_type == 'ADMG':
            data = pickle.load(open(hidden_true_data, 'rb'))


if __name__ == '__main__':
    unit_test()

