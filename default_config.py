import os
import argparse
import sys
import datetime
import logging

process_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data_preprocess'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'model'))
t = datetime.datetime.now()
time = ".".join([str(t.year), str(t.month), str(t.day), str(t.hour), str(t.minute), str(t.second)])
script_path = os.path.split(os.path.realpath(__file__))[0]
resource_folder = os.path.join(script_path, 'resource')
data_folder = os.path.join(script_path, 'resource', 'data')
adjacency_mat_folder = os.path.join(script_path, 'resource', 'adjacency_mat_folder')
ckpt_folder = os.path.join(script_path, 'resource', 'ckpt_folder')
treatment_result_inference_folder = os.path.join(script_path, 'resource', 'treatment_result_inference')
fig_save_folder = os.path.join(script_path, 'resource', 'figure')

dataset = 'hao_true_lmci' # 'zheng
hidden_flag = 'True' # False
device = 'cuda:0'
constraint_type = 'DAG'
model = 'ODE'
prior_causal_mask = 'not_causal' # hao_true_causal, not_causal use_data
non_linear_mode = "True"
treatment_optimize_method = 'difference' # difference min
new_model_number = 4
treatment_filter_threshold = 0.04


# hao_true_lmci
# treatment_feature = 'n'
# treatment_time = 52
# treatment_observation_time = 57
# treatment_value = 0

# zheng
# treatment_feature = 'n'
# treatment_time = 0
# treatment_observation_time = 10
# treatment_value = 0

# auto
treatment_feature = 'node_15'
treatment_time = 1
treatment_observation_time = 3
treatment_value = 1
assert model in {'ODE'}

# treatment_init_model_name = 'predict.CTP.hao_true_lmci.True.none.20230404121230308383.18.2900.model'
# treatment_init_model_name = 'predict.CTP.zheng.False.DAG.20230406082853239080.8.100.model'
treatment_init_model_name = 'predict.CTP.auto50.True.DAG.20230406104408175186.99.100.model'


if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(adjacency_mat_folder):
    os.makedirs(adjacency_mat_folder)
if not os.path.exists(ckpt_folder):
    os.makedirs(ckpt_folder)
if not os.path.exists(treatment_result_inference_folder):
    os.makedirs(treatment_result_inference_folder)
if not os.path.exists(fig_save_folder):
    os.makedirs(fig_save_folder)
missing_flag_num = -99999


default_config = {
    'process_name': process_name,
    'model': model,
    'prior_causal_mask': prior_causal_mask,
    # 'causal_derivative_flag': causal_derivative_flag,

    # dataset config
    'dataset_name': dataset,
    "data_path": None,
    "batch_first": "True",
    "minimum_observation": None,
    "input_size": None,
    "mask_tag": missing_flag_num,
    "reconstruct_input": "True",
    "predict_label": "True",
    'time_offset': None,
    'distribution_mode': 'None',
    'hidden_flag': hidden_flag,

    # model config
    "hidden_size": 8,
    'init_pooling': 'mean',
    'init_net_bidirectional': "True",
    'non_linear_mode': non_linear_mode,

    # train setting
    'max_epoch': 3000,
    'max_iteration': 3000,
    "batch_size": 128,
    "model_converge_threshold": 10**-8,
    "clamp_edge_threshold": 10**-4,
    "learning_rate": 0.001,
    "eval_iter_interval": 20,
    "eval_epoch_interval": -1,
    "device": device,
    "clamp_edge_flag": "False",
    'adjacency_mat_folder': adjacency_mat_folder,
    'save_iter_interval': 100,

    # graph setting
    "constraint_type": constraint_type,
    'sparse_constraint_weight': 0,

    # treatment
    'treatment_feature': treatment_feature,
    'treatment_time': treatment_time,
    'treatment_observation_time': treatment_observation_time,
    'treatment_value': treatment_value,
    'treatment_sample_multiplier': 1,
    'treatment_eval_iter_interval': 20,
    "treatment_predict_lr" : 0.0001,
    "treatment_treatment_lr" : 0.000003,
    "treatment_max_epoch" : 200,
    "treatment_max_iter" : 2000,
    'treatment_new_model_number': new_model_number,
    'treatment_warm_iter': 200,
    'treatment_filter_threshold': treatment_filter_threshold,
    'treatment_optimize_method': treatment_optimize_method,
    'treatment_random_observation_time': 'False',
    'treatment_true_causal': 'True',

    # augmented Lagrangian predict phase
    "init_lambda_predict": 0.0,
    "init_mu_predict": 10**-3,
    "max_lambda_predict": 0,
    "max_mu_predict": 0,
    "eta_predict": 10,
    'gamma_predict': 0.9,
    'stop_threshold_predict': 10**-8,
    'update_window_predict': 50,
    'lagrangian_converge_threshold_predict': 10**-2,

    # augmented Lagrangian treatment effect analysis phase
    "init_lambda_treatment": 0,
    "init_mu_treatment": 10**-3,
    "eta_treatment": 10,
    'gamma_treatment': 0.9,
    'stop_threshold_treatment': 10**-8,
    'update_window_treatment': 50,
    'lagrangian_converge_threshold_treatment': 10**-2,
}


parser = argparse.ArgumentParser()
parser.add_argument('--process_name', help='', default=default_config['process_name'], type=str)
parser.add_argument('--model_name', help='', default=default_config['model'], type=str)
parser.add_argument('--prior_causal_mask', help='', default=default_config['prior_causal_mask'], type=str)
# parser.add_argument('--causal_derivative_flag', help='', default=default_config['causal_derivative_flag'], type=str)

# dataset config
parser.add_argument('--dataset_name', help='', default=default_config['dataset_name'], type=str)
parser.add_argument('--data_path', help='', default=default_config['data_path'], type=str)
parser.add_argument('--batch_first', help='', default=default_config['batch_first'], type=str)
parser.add_argument('--minimum_observation', help='', default=default_config['minimum_observation'], type=int)
parser.add_argument('--input_size', help='', default=default_config['input_size'], type=int)
parser.add_argument('--mask_tag', help='', default=default_config['mask_tag'], type=int)
parser.add_argument('--reconstruct_input', help='', default=default_config['reconstruct_input'], type=str)
parser.add_argument('--predict_label', help='', default=default_config['predict_label'], type=str)
parser.add_argument('--time_offset', help='', default=default_config['time_offset'], type=int)
parser.add_argument('--distribution_mode', help='', default=default_config['distribution_mode'], type=str)
parser.add_argument('--hidden_flag', help='', default=default_config['hidden_flag'], type=str)

# model config
parser.add_argument('--non_linear_mode', help='', default=default_config['non_linear_mode'], type=str)
parser.add_argument('--hidden_size', help='', default=default_config['hidden_size'], type=int)
parser.add_argument('--init_pooling', help='', default=default_config['init_pooling'], type=str)
parser.add_argument('--init_net_bidirectional', help='',
                    default=default_config['init_net_bidirectional'], type=str)

# training setting
parser.add_argument('--batch_size', help='', default=default_config['batch_size'], type=int)
parser.add_argument('--device', help='', default=default_config['device'], type=str)
parser.add_argument('--max_epoch', help='', default=default_config['max_epoch'], type=int)
parser.add_argument('--clamp_edge_flag', help='', default=default_config['clamp_edge_flag'], type=str)
parser.add_argument('--adjacency_mat_folder', help='', default=default_config['adjacency_mat_folder'], type=str)
parser.add_argument('--clamp_edge_threshold', help='', default=default_config['clamp_edge_threshold'], type=float)
parser.add_argument('--max_iteration', help='', default=default_config['max_iteration'], type=int)
parser.add_argument('--learning_rate', help='', default=default_config['learning_rate'], type=float)
parser.add_argument('--model_converge_threshold', help='',
                    default=default_config['model_converge_threshold'], type=float)
parser.add_argument('--eval_iter_interval', help='', default=default_config['eval_iter_interval'], type=str)
parser.add_argument('--eval_epoch_interval', help='', default=default_config['eval_epoch_interval'], type=str)
parser.add_argument('--save_iter_interval', help='', default=default_config['save_iter_interval'], type=int)
parser.add_argument('--log_every_iteration', help='', default='False', type=str)

# graph setting
parser.add_argument('--constraint_type', help='', default=default_config['constraint_type'], type=str)
parser.add_argument('--sparse_constraint_weight', help='', default=default_config['sparse_constraint_weight'],
                    type=float)

# augmented Lagrangian predict
parser.add_argument('--init_lambda_predict', help='', default=default_config['init_lambda_predict'], type=float)
parser.add_argument('--init_mu_predict', help='', default=default_config['init_mu_predict'], type=float)
parser.add_argument('--max_lambda_predict', help='', default=default_config['max_lambda_predict'], type=float)
parser.add_argument('--max_mu_predict', help='', default=default_config['max_mu_predict'], type=float)
parser.add_argument('--eta_predict', help='', default=default_config['eta_predict'], type=float)
parser.add_argument('--gamma_predict', help='', default=default_config['gamma_predict'], type=float)
parser.add_argument('--stop_threshold_predict', help='', default=default_config['stop_threshold_predict'], type=float)
parser.add_argument('--update_window_predict', help='', default=default_config['update_window_predict'], type=float)
parser.add_argument('--lagrangian_converge_threshold_predict', help='',
                    default=default_config['lagrangian_converge_threshold_predict'], type=float)

# treatment analysis
parser.add_argument('--treatment_init_model_name', help='', default=treatment_init_model_name, type=str)
parser.add_argument('--treatment_warm_iter', help='', default=default_config['treatment_warm_iter'], type=int)
parser.add_argument('--treatment_feature', help='', default=default_config['treatment_feature'], type=str)
parser.add_argument('--treatment_time', help='', default=default_config['treatment_time'], type=float)
parser.add_argument('--treatment_value', help='', default=default_config['treatment_value'], type=float)
parser.add_argument('--treatment_random_observation_time', help='',
                    default=default_config['treatment_random_observation_time'], type=str)
parser.add_argument('--treatment_sample_multiplier', help='', default=default_config['treatment_sample_multiplier'],
                    type=int)
parser.add_argument('--treatment_optimize_method', help='',
                    default=default_config['treatment_optimize_method'], type=str)

parser.add_argument('--treatment_predict_lr', help='', default=default_config['treatment_predict_lr'], type=float)
parser.add_argument('--treatment_treatment_lr', help='', default=default_config['treatment_treatment_lr'], type=float)
parser.add_argument('--treatment_observation_time', help='', default=default_config['treatment_observation_time'],
                    type=float)
parser.add_argument('--treatment_max_iter', help='', default=default_config['treatment_max_iter'], type=int)
parser.add_argument('--treatment_max_epoch', help='', default=default_config['treatment_max_epoch'], type=int)
parser.add_argument('--treatment_eval_iter_interval', help='', default=default_config['treatment_eval_iter_interval'],
                    type=int)
parser.add_argument('--treatment_new_model_number', help='', default=default_config['treatment_new_model_number'],
                    type=int)
parser.add_argument('--treatment_filter_threshold', help='', default=default_config['treatment_filter_threshold'],
                    type=float)
parser.add_argument('--treatment_true_causal', help='', default=default_config['treatment_true_causal'],
                    type=str)


args = vars(parser.parse_args())
if args["dataset_name"] == 'hao_true_lmci':
    distribution_mode = 'uniform'
    args["data_path"] = os.path.join(data_folder, 'sim_hao_model_hidden_True_personal_2_type_{}.pkl'
                                     .format(distribution_mode))
    args["time_offset"] = 50
    args["minimum_observation"] = 4
    args["input_size"] = 4
    args["hidden_size"] = 8
    args["sparse_constraint_weight"] = 0.08
    args["max_lambda_predict"] = 0.5
    args["max_mu_predict"] = 2
    args["distribution_mode"] = distribution_mode
elif args["dataset_name"] == 'hao_false_lmci':
    distribution_mode = 'uniform'
    args["data_path"] = os.path.join(data_folder, 'sim_hao_model_hidden_False_personal_2_type_{}.pkl'
                                     .format(distribution_mode))
    args["time_offset"] = 50
    args["minimum_observation"] = 4
    args["input_size"] = 5
    args["hidden_size"] = 8
    args["sparse_constraint_weight"] = 0.08
    args["max_lambda_predict"] = 0.5
    args["max_mu_predict"] = 2
    args["distribution_mode"] = distribution_mode
elif args["dataset_name"] == 'spiral_2d':
    distribution_mode = 'uniform'
    args["data_path"] = os.path.join(data_folder, 'spiral_2d.pkl')
    args["minimum_observation"] = 100
    args["time_offset"] = 0
    args["input_size"] = 2
    args["hidden_size"] = 2
    args["sparse_constraint_weight"] = 0.08
    args["max_lambda_predict"] = 0.5
    args["max_mu_predict"] = 2
    args["distribution_mode"] = distribution_mode
elif args["dataset_name"] == 'zheng':
    distribution_mode = 'uniform'
    args["data_path"] = os.path.join(data_folder, 'sim_zheng_model_hidden_True_personal_2_type_{}.pkl'
                                     .format(distribution_mode))
    args["time_offset"] = -10
    args["minimum_observation"] = 4
    args["input_size"] = 4
    args["hidden_size"] = 8
    args["sparse_constraint_weight"] = 2
    args["max_lambda_predict"] = 100
    args["max_mu_predict"] = 200
    args["distribution_mode"] = distribution_mode
elif args["dataset_name"] == 'auto25':
    distribution_mode = 'uniform'
    args["data_path"] = os.path.join(data_folder, 'sim_auto25_model_hidden_True_personal_2_type_{}.pkl'
                                     .format(distribution_mode))
    args["time_offset"] = 0
    args["minimum_observation"] = 4
    args["input_size"] = 20
    args["hidden_size"] = 16
    args["sparse_constraint_weight"] = 2
    args["max_lambda_predict"] = 100
    args["max_mu_predict"] = 200
    args["distribution_mode"] = distribution_mode
elif args["dataset_name"] == 'auto50':
    distribution_mode = 'uniform'
    args["data_path"] = os.path.join(data_folder, 'sim_auto50_model_hidden_True_personal_2_type_{}.pkl'
                                     .format(distribution_mode))
    args["time_offset"] = 0
    args["minimum_observation"] = 4
    args["input_size"] = 45
    args["hidden_size"] = 32
    args["sparse_constraint_weight"] = 2
    args["max_lambda_predict"] = 100
    args["max_mu_predict"] = 200
    args["distribution_mode"] = distribution_mode
elif args["dataset_name"] == 'adni':
    distribution_mode = 'random'
    args["data_path"] = os.path.join(data_folder, 'ADNI_merge_preprocessed.pkl')
    args["time_offset"] = 0
    args["minimum_observation"] = 2
    args["log_every_iteration"] = 'True'
    args["input_size"] = 88
    args["hidden_size"] = 32
    args['batch_size'] = 32
    args["sparse_constraint_weight"] = 2
    args["max_lambda_predict"] = 100
    args["max_mu_predict"] = 200
    args["distribution_mode"] = distribution_mode
else:
    raise ValueError('')


# logger
log_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resource', 'log')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_file_name = os.path.join(log_folder, '{}.txt'.format('log'))
format_ = "%(asctime)s %(process)d %(module)s %(lineno)d %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format=format_, filename=log_file_name)
console_logger = logging.StreamHandler()
# console output format
stream_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")
# file output format
console_logger.setFormatter(stream_format)
logger.addHandler(console_logger)
config_list = []
for key in args:
    config_list.append([key, args[key]])
config_list = sorted(config_list, key=lambda x: x[0])
for item in config_list:
    logger.info("{}: {}".format(item[0], item[1]))
