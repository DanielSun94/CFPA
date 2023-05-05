import os
import argparse
import sys
import datetime
import logging


sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data_preprocess'))
t = datetime.datetime.now()
time = ".".join([str(t.year), str(t.month), str(t.day), str(t.hour), str(t.minute), str(t.second)])
data_folder = os.path.abspath('../resource/simulated_data/')
dataset = 'spiral_2d'

if dataset == 'hao_true':
    data_path = os.path.join(data_folder, 'sim_hao_model_hidden_True_group_lmci_personal_2_type_random.pkl')
    time_offset = 50
    minimum_observation = 4
    input_size = 4
elif dataset == 'hao_false':
    data_path = os.path.join(data_folder, 'sim_hao_model_hidden_False_group_lmci_personal_2_type_random.pkl')
    time_offset = 50
    minimum_observation = 4
    input_size = 5
elif dataset == 'spiral_2d':
    data_path = os.path.join(data_folder, 'spiral_2d.pkl')
    minimum_observation = 102
    time_offset = 0
    input_size = 2
else:
    raise ValueError('')
missing_flag_num = -99999


default_config = {
    'process_name': 'verification',

    # dataset config
    'dataset_name': dataset,
    "data_path": data_path,
    "batch_first": "True",
    "minimum_observation": minimum_observation,
    "input_size": input_size,
    "mask_tag": missing_flag_num,
    "reconstruct_input": "True",
    "predict_label": "True",
    'time_offset': time_offset,

    # model config
    "mediate_size": 2,
    "hidden_size": 4,

    # train setting
    'max_epoch': 10000,
    'max_iteration': 1000000,
    "batch_size": 128,
    "model_converge_threshold": 10**-8,
    "learning_rate": 0.01,
    "eval_iter_interval": 5,
    "eval_epoch_interval": -1,

    # graph setting
    "constraint_type": 'ancestral',  # valid value: ancestral, arid, bow-free (for ADMG), and default (for DAG)
    'graph_type': 'ADMG',  # valid value: ADMG, DAG

    # augmented Lagrangian
    "init_lambda": 0,
    "init_mu": 10**-3,
    "eta": 10,
    'gamma': 0.9,
    'stop_threshold': 10**-8,
    'update_window': 50,
    'lagrangian_converge_threshold': 10**-4
}

parser = argparse.ArgumentParser()
parser.add_argument('--process_name', help='', default=default_config['process_name'], type=str)

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

# model config
parser.add_argument('--mediate_size', help='', default=default_config['mediate_size'], type=int)
parser.add_argument('--hidden_size', help='', default=default_config['hidden_size'], type=int)

# training setting
parser.add_argument('--batch_size', help='', default=default_config['batch_size'], type=int)
parser.add_argument('--max_epoch', help='', default=default_config['max_epoch'], type=int)
parser.add_argument('--max_iteration', help='', default=default_config['max_iteration'], type=int)
parser.add_argument('--learning_rate', help='', default=default_config['learning_rate'], type=float)
parser.add_argument('--model_converge_threshold', help='',
                    default=default_config['model_converge_threshold'], type=float)
parser.add_argument('--eval_iter_interval', help='', default=default_config['eval_iter_interval'], type=str)
parser.add_argument('--eval_epoch_interval', help='', default=default_config['eval_epoch_interval'], type=str)

# graph setting
parser.add_argument('--constraint_type', help='', default=default_config['constraint_type'], type=str)
parser.add_argument('--graph_type', help='', default=default_config['graph_type'], type=str)

# augmented Lagrangian
parser.add_argument('--init_lambda', help='', default=default_config['init_lambda'], type=float)
parser.add_argument('--init_mu', help='', default=default_config['init_mu'], type=float)
parser.add_argument('--eta', help='', default=default_config['eta'], type=float)
parser.add_argument('--gamma', help='', default=default_config['gamma'], type=float)
parser.add_argument('--stop_threshold', help='', default=default_config['stop_threshold'], type=float)
parser.add_argument('--update_window', help='', default=default_config['update_window'], type=float)
parser.add_argument('--lagrangian_converge_threshold', help='',
                    default=default_config['lagrangian_converge_threshold'], type=float)

args = vars(parser.parse_args())


# logger
log_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resource', 'log')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_file_name = os.path.join(log_folder, '{}.txt'.format(args['process_name']))
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

# other config

