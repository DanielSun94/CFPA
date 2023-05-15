import os
from default_config import args as argument, ckpt_folder
from torch.nn import Module
from torch import load, zeros_like, FloatTensor
import numpy as np



class TreatmentEffectEstimator(Module):
    def __init__(self, model_ckpt_path, dataset_name, treatment_feature, treatment_time, treatment_value, device,
                 name_id_dict, oracle_graph=None):
        super().__init__()
        model = load(model_ckpt_path)

        # input check
        assert isinstance(name_id_dict, dict) and isinstance(treatment_feature, str) and \
               (treatment_feature in name_id_dict)
        # treatment time为-1，-2时表示施加干预的时间是最后一次入院的时间/倒数第二次入院的时间，是正数时代表任意指定时间
        assert treatment_time == -1 or treatment_time == -2 or \
               (treatment_time > 0 and isinstance(treatment_time, float))
        # treatment value -1, -2代表施加干预的值即为最后一次入(倒数第二次)院的指标观测值，正数时代表任意指定值
        assert treatment_value == -1 or treatment_value == -2 \
               or (treatment_value > 0 and isinstance(treatment_value, float))
        assert dataset_name == model.dataset_name
        self.model = model.to(device)
        self.treatment_feature = treatment_feature
        self.treatment_idx = name_id_dict[treatment_feature]
        self.treatment_value = self.treatment_value
        self.treatment_time = self.treatment_time
        self.name_id_dict = name_id_dict
        self.dataset_name = dataset_name
        adjacency = model.causal_derivative.adjacency
        dag = adjacency['dag']
        bi = adjacency['bi'] if 'bi' in adjacency else zeros_like(dag)
        adjacency = ((dag + bi) > 0).float()
        if oracle_graph is None:
            self.adjacency = adjacency
        else:
            self.adjacency = self.__oracle_graph_reformat(oracle_graph)
        self.device = device


    def reconstruct_adjacency_mat(self):



    def __oracle_graph_reformat(self, oracle_graph):
        name_idx_dict= self.name_id_dict
        adjacency = np.zeros([len(name_idx_dict), len(name_idx_dict)])
        for key_1 in oracle_graph:
            for key_2 in oracle_graph[key_1]:
                value = oracle_graph[key_1][key_2]
                if value > 0:
                    idx_1 = name_idx_dict[key_1]
                    idx_2 = name_idx_dict[key_2]
                    adjacency[idx_1, idx_2] = 1
        adjacency = FloatTensor(adjacency)
        return adjacency


    def re_fit(self, data):
        treatment_idx = self.treatment_idx
        treatment_value = self.treatment_value
        treatment_time = self.treatment_time


def main():
    path = 'CPA.hao_false.DAG.default.20230407075131.0.1.model'
    model_path = os.path.join(ckpt_folder, path)


if __name__ == '__main__':
    main()