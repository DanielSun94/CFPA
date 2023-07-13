from default_config import treatment_result_inference_folder
import numpy as np
import os
import csv


def read_treatment_data(data_name, feature_list):
    data_path = os.path.join(treatment_result_inference_folder, data_name)
    time_list = []
    data_dict = dict()
    with open(data_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            if i == 0:
                for j in range(4, len(line)):
                    time_list.append(float(line[j]))
                continue
            sample_id, model, feature, data_type = line[0: 4]
            data = np.array([float(item) for item in line[4:]])
            if sample_id not in data_dict:
                data_dict[sample_id] = dict()
            if model not in data_dict[sample_id]:
                data_dict[sample_id][model] = dict()
            if feature not in data_dict[sample_id][model] and feature in feature_list:
                data_dict[sample_id][model][feature] = dict()
            data_dict[sample_id][model][feature][data_type] = data
    return data_dict, time_list