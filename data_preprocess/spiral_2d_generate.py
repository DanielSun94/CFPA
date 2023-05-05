import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import pickle
import os


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=105,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=False):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral， 只使用前N个数据
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns:
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


def main():
    _, train_traj, _, train_time = generate_spiral2d(nspiral=1024)
    _, val_traj, _, val_time = generate_spiral2d(nspiral=128)
    _, test_traj, _, test_time = generate_spiral2d(nspiral=128)
    train_dataset, val_dataset, test_dataset = [], [], []
    train_data_list, val_data_list, test_data_list = [], [], []

    for traj_list, time_list, dataset in zip([train_traj, val_traj, test_traj], [train_time, val_time, test_time],
                                             [train_dataset, val_dataset, test_dataset]):
        for traj in traj_list:
            sample = []
            for visit, time_point in zip(traj, time_list):
                sample.append({'x': visit[0], 'y': visit[1], 'visit_time': time_point})
            dataset.append(sample)

    for data, data_list in zip([train_dataset, val_dataset, test_dataset],
                               [train_data_list, val_data_list, test_data_list]):
        for item in data:
            sample = {'observation': item, 'true_value': item}
            data_list.append(sample)

    save_name = 'spiral_2d.pkl'
    default_save_data_folder = os.path.abspath('../resource/simulated_data')
    pickle.dump(
        {
            'data': {
                'train': train_data_list,
                'valid': val_data_list,
                'test': test_data_list,
            },
        },
        open(os.path.join(default_save_data_folder, save_name), 'wb')
    )


if __name__ == '__main__':
    main()
