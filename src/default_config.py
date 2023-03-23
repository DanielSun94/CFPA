import os

simulated_data_folder = os.path.abspath('../resource/simulated_data')
simulated_data_config_path = os.path.abspath('../resource/simulated_data_config.yaml')
if not os.path.exists(simulated_data_folder):
    os.makedirs(simulated_data_folder)
if not os.path.exists(simulated_data_config_path):
    raise ValueError('')
