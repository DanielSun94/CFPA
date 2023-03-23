import os
import random
import argparse
import pickle
from scipy.integrate import solve_ivp
from yaml import Loader, load


def main():
    default_save_data_folder = os.path.abspath('../../resource/simulated_data')
    default_config_path = os.path.abspath('../../resource/hao_model_config.yaml')
    default_use_hidden = "False"
    default_group = 'lmci'
    default_sample_type = 'random'
    default_train_sample_size = 800
    default_valid_sample_size = 200
    default_test_sample_size = 200
    default_personalized_type = 0
    parser = argparse.ArgumentParser(description='simulate data generating')
    parser.add_argument('--config_path', type=str, default=default_config_path)
    parser.add_argument('--use_hidden', type=str, default=default_use_hidden)
    parser.add_argument('--group', type=str, default=default_group)
    parser.add_argument('--sample_type', type=str, default=default_sample_type)
    parser.add_argument('--train_sample_size', type=int, default=default_train_sample_size)
    parser.add_argument('--valid_sample_size', type=int, default=default_valid_sample_size)
    parser.add_argument('--test_sample_size', type=int, default=default_test_sample_size)
    parser.add_argument('--personalized_type', type=int, default=default_personalized_type)
    parser.add_argument('--save_data_folder', type=str, default=default_save_data_folder)
    args = parser.parse_args()

    setting_dict = {
        'config_path': args.config_path,
        'use_hidden': args.use_hidden,
        'group': args.group,
        'sample_type': args.sample_type,
        'train_sample_size': args.train_sample_size,
        'valid_sample_size': args.valid_sample_size,
        'test_sample_size': args.test_sample_size,
        'personalized_type': args.personalized_type,
        'save_data_folder': args.save_data_folder,
    }

    if not os.path.exists(setting_dict['save_data_folder']):
        os.makedirs(setting_dict['save_data_folder'])

    config_path = setting_dict['config_path']
    use_hidden = setting_dict['use_hidden']
    train_sample_size = setting_dict['train_sample_size']
    valid_sample_size = setting_dict['valid_sample_size']
    test_sample_size = setting_dict['test_sample_size']
    personalized_type = setting_dict['personalized_type']
    group = setting_dict['group']
    sample_type = setting_dict['sample_type']

    config = load(open(config_path, 'r'), Loader)
    model = HaoModel(config, use_hidden)

    test_data = model.generate_dataset(test_sample_size, personalized_type, group, sample_type)
    train_data = model.generate_dataset(train_sample_size, personalized_type, group, sample_type)
    valid_data = model.generate_dataset(valid_sample_size, personalized_type, group, sample_type)

    save_name = 'sim_data_hidden_{}_group_{}_personal_{}_type_{}.pkl'\
        .format(use_hidden, group, personalized_type, sample_type)
    pickle.dump(
        {
            'config': config,
            'setting': setting_dict,
            'data': {
                'train': train_data,
                'valid': valid_data,
                'test': test_data
            }
        },
        open(os.path.join(setting_dict['save_data_folder'], save_name), 'wb')
    )


class HaoModel(object):
    """
    Model Reference
    Hao, Wenrui, Suzanne Lenhart, and Jeffrey R. Petrella. "Optimal anti-amyloid-beta therapy for Alzheimer’s
    disease via a personalized mathematical model." PLoS computational biology 18.9 (2022): e1010481.
    """
    def __init__(self, config, hidden):
        self.__missing_rate_dict = {}
        self.__sample_info = {}
        self.__ad_init_mean, self.__ad_init_std = {}, {}
        self.__lmci_init_mean, self.__lmci_init_std = {}, {}
        self.__cn_init_mean, self.__cn_init_std = {}, {}
        self.__ad_para_mean, self.__ad_para_std = {}, {}
        self.__lmci_para_mean, self.__lmci_para_std = {}, {}
        self.__cn_para_mean, self.__cn_para_std = {}, {}
        self.__load_config(config)
        self.__use_hidden = hidden

    def __load_config(self, config):
        miss_dict = config['missing_rate']
        sample = config['sample_characteristic']
        for key in miss_dict:
            self.__missing_rate_dict[key] = miss_dict[key]
        self.__sample_info['t_0'] = sample['t_0']
        self.__sample_info['random_max_visit'] = sample['random_max_visit']
        self.__sample_info['random_min_visit'] = sample['random_min_visit']
        self.__sample_info['random_max_interval'] = sample['random_max_interval']
        self.__sample_info['random_min_interval'] = sample['random_min_interval']
        self.__sample_info['uniform_visit'] = sample['uniform_visit']
        self.__sample_info['uniform_interval'] = sample['uniform_interval']
        self.__sample_info['personalized_turb_1'] = sample['personalized_turbulence_coefficient_1']
        self.__sample_info['personalized_turb_2'] = sample['personalized_turbulence_coefficient_2']
        self.__sample_info['noise_coefficient'] = sample['gaussian_noise_std_coefficient']
        self.__sample_info['unit'] = sample['unit']
        self.__sample_info['uniform'] = sample['distribution']
        for ((mean, std), info_dict) in \
            zip([[self.__ad_para_mean, self.__ad_para_std], [self.__lmci_para_mean, self.__lmci_para_std],
                 [self.__cn_para_mean, self.__cn_para_std]],
                [config['ad']['parameters'], config['lmci']['parameters'], config['cn']['parameters']]):
            for key in info_dict:
                mean[key] = info_dict[key]['mean']
                std[key] = info_dict[key]['std']
        for ((mean, std), info_dict) in \
            zip([[self.__ad_init_mean, self.__ad_init_std], [self.__lmci_init_mean, self.__lmci_init_std],
                 [self.__cn_init_mean, self.__cn_init_std]],
                [config['ad']['base_initial_value'], config['lmci']['base_initial_value'],
                 config['cn']['base_initial_value']]):
            for key in info_dict:
                mean[key] = info_dict[key]['mean']
                std[key] = info_dict[key]['std']

    def __generate_trajectory(self, init_time, intervals, init_mean, init_std, para_mean, para_std, personalized):
        init, para = self.__personalized_parameter_generating(init_mean, init_std, para_mean, para_std, personalized)
        t_init = self.__sample_info['t_0']
        trajectory = {'init': init, 'para': para, 'observation': [], 'true_value': []}
        for item in intervals:
            visit_time = item + init_time
            state = self.__calculate_state(init, para, visit_time, t_init)
            observed_state = self.__add_noise(state, init_mean)
            observed_state['visit_time'] = visit_time
            state['visit_time'] = visit_time
            trajectory['observation'].append(observed_state)
            trajectory['true_value'].append(state)
        return trajectory

    def __calculate_state(self, init, para, visit_time, t_init):
        """
        calculate the model in a cascade manner, i.e., the a_eta, tau_p, tau_o, n, c
        the ode order strictly follow the page 5 of the paper.
        :param init:
        :param para:
        :param visit_time:
        :param t_init:
        :return:
        """
        use_hidden = self.__use_hidden
        a_init, tau_p_init, tau_o_init, n_init, c_init = init['a'], init['tau_p'], init['tau_o'], init['n'], init['c']
        lambda_a_beta = para['lambda_a_beta']
        k_a_beta = para['k_a_beta']
        lambda_tau = para['lambda_tau']
        k_tau = para['k_tau']
        lambda_tau_o = para['lambda_tau_o']
        lambda_ntau_o = para['lambda_ntau_o']
        lambda_ntau_p = para['lambda_ntau_p']
        k_n = para['k_n']
        lambda_cn = para['lambda_cn']
        lambda_ctau = para['lambda_ctau']
        k_c = para['k_c']

        def hao_dynamic_system(_, y):
            """
            if use hidden is true, we presume tau_o also influence the cognitive ability
            the tau_o -> c connection (as well as the parameter) is purely hypothetical. It is only used for the
            confounder identification test
            """
            if use_hidden == "True":
                return [
                    lambda_a_beta * y[0] * (1 - y[0] / k_a_beta),
                    lambda_tau * y[0] * (1 - y[1] / k_tau),
                    lambda_tau_o,
                    (lambda_ntau_o * y[2] + lambda_ntau_p * y[1]) * (1 - y[3] / k_n),
                    (lambda_cn * y[3] + 0.4 * lambda_ctau * (y[1] + y[2])) * (1 - y[4] / k_c)
                ]
            elif use_hidden == "False":
                return [
                    lambda_a_beta * y[0] * (1 - y[0] / k_a_beta),
                    lambda_tau * y[0] * (1 - y[1] / k_tau),
                    lambda_tau_o,
                    (lambda_ntau_o * y[2] + lambda_ntau_p * y[1]) * (1 - y[3] / k_n),
                    (lambda_cn * y[3] + lambda_ctau * y[1]) * (1 - y[4] / k_c)
                ]
            else:
                raise ValueError('')

        t_span = t_init, visit_time
        initial_state = [a_init, tau_p_init, tau_o_init, n_init, c_init]
        full_result = solve_ivp(hao_dynamic_system, t_span, initial_state)
        result = full_result.y[:, -1]
        return {
            'a': result[0],
            'tau_p': result[1],
            'tau_o': result[2],
            'n': result[3],
            'c': result[4],
        }

    def __personalized_parameter_generating(self, init_mean, init_std, para_mean, para_std, personalized):
        """
        personalized 0, all patients share the same init value and para
        personalized 1: all patients share the same init value, parameters varies
        personalized 2：all patients share the same parameters, init value varies
        personalized 3: both init and parameter vary
        """
        assert isinstance(personalized, int) and 0 <= personalized <= 3
        turb_1 = self.__sample_info['personalized_turb_1']
        turb_2 = self.__sample_info['personalized_turb_2']
        new_para, new_init = {}, {}
        for key in para_mean:
            key_para_mean = para_mean[key]
            key_para_std = para_std[key]
            if key_para_mean > key_para_std:
                para_turb_range = key_para_std * random.uniform(-turb_1, turb_1)
            else:
                para_turb_range = key_para_mean * random.uniform(-turb_2, turb_2)
            new_para[key] = key_para_mean + para_turb_range
        for key in init_mean:
            key_init_mean = init_mean[key]
            key_init_std = init_std[key]
            if key_init_mean > key_init_std:
                init_turb_range = key_init_std * random.uniform(-turb_1, turb_1)
            else:
                init_turb_range = key_init_mean * random.uniform(-turb_2, turb_2)
            new_init[key] = key_init_mean + init_turb_range

        if personalized == 0:
            return init_mean, para_mean
        elif personalized == 1:
            return init_mean, new_para
        elif personalized == 2:
            return new_init, para_mean
        elif personalized == 3:
            return new_init, new_para
        else:
            raise ValueError('')

    def __add_noise(self, sample_dict, init_mean):
        """
        There are two types of noise, the first is observation noise which occurs in every feature.
        The noise signal follows a Gaussian distribution with mean zero and a standard variance
        The standard variance is the gaussian_ratio * init value
        The sampled data will be discarded (and resampled) if it is a negative number
        The gaussian ratio need to be less than 1/3 to avoid negative value sample as much as possible
        (and keep the sampled data follows Gaussian distribution)

        The second type is the missing noise, which means we lost signal with respect to a preset lost ratio
        """
        noisy_sample = {}
        for key in sample_dict:
            noise = -10**6
            while -1 * noise > sample_dict[key]:
                standard_variance = init_mean[key] * self.__sample_info['noise_coefficient']
                noise = random.gauss(0, standard_variance)
                if key in noisy_sample:
                    print('previous sample failed')
                noisy_sample[key] = noise + sample_dict[key]

        missing_rate = self.__missing_rate_dict
        final_sample = {}
        for key in missing_rate:
            threshold = missing_rate[key]
            ran_num = random.uniform(0, 1)
            if ran_num > threshold:
                final_sample[key] = noisy_sample[key]
            else:
                final_sample[key] = -1
        return final_sample

    def generate_dataset(self, sample_size, personalized_type, group, sample_type):
        dataset = []
        random_min_visit = self.__sample_info['random_min_visit']
        random_max_visit = self.__sample_info['random_max_visit']
        random_max_interval = self.__sample_info['random_max_interval']
        random_min_interval = self.__sample_info['random_min_interval']
        uniform_interval = self.__sample_info['uniform_interval']
        uniform_visit = self.__sample_info['uniform_visit']
        init_time = self.__sample_info['t_0']
        init_mean, init_std, para_mean, para_std = self.__get_generating_info(group)
        for i in range(sample_size):
            if sample_type == 'random':
                visit_num = random.randint(random_min_visit, random_max_visit)
                visit_interval_list = [0]
                for j in range(visit_num):
                    visit_interval_list.append(random.uniform(random_min_interval, random_max_interval) +
                                               visit_interval_list[-1])
                visit_interval_list = visit_interval_list[1:]
            elif sample_type == 'uniform':
                visit_num = uniform_visit
                visit_interval_list = [0]
                for j in range(visit_num):
                    visit_interval_list.append(uniform_interval + visit_interval_list[-1])
                visit_interval_list = visit_interval_list[1:]
            else:
                raise ValueError('')
            trajectory = self.__generate_trajectory(init_time, visit_interval_list, init_mean, init_std, para_mean,
                                                    para_std, personalized_type)
            dataset.append(trajectory)
        return dataset

    def __get_generating_info(self, group):
        if group == 'lmci':
            init_mean = self.__lmci_init_mean
            init_std = self.__lmci_init_std
            para_mean = self.__lmci_para_mean
            para_std = self.__lmci_para_std
        elif group == 'ad':
            init_mean = self.__ad_init_mean
            init_std = self.__ad_init_std
            para_mean = self.__ad_para_mean
            para_std = self.__ad_para_std
        elif group == 'cn':
            init_mean = self.__cn_init_mean
            init_std = self.__cn_init_std
            para_mean = self.__cn_para_mean
            para_std = self.__cn_para_std
        else:
            raise ValueError('')
        return init_mean, init_std, para_mean, para_std


if __name__ == '__main__':
    main()
