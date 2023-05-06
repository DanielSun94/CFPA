from torch import nn
import torch
import numpy as np
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from default_config import args as argument
from util import get_data_loader


def main():
    loss_func_name = 'MSE'
    print('loss_func_name: {}'.format(loss_func_name))
    mse_loss_func = torch.nn.MSELoss()
    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    noise_std = .3
    device = torch.device('cpu')

    dataset_name = argument['dataset_name']
    data_path = argument['data_path']
    learning_rate = argument['learning_rate']
    reconstruct_input = True if argument['reconstruct_input'] == 'True' else False
    predict_label = True if argument['predict_label'] == 'True' else False
    mask_tag = argument['mask_tag']
    minimum_observation = argument['minimum_observation']
    batch_size = 1000


    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_size).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=learning_rate)
    loss_meter_kl, loss_meter_mse = RunningAverageMeter(), RunningAverageMeter()

    dataloader_dict = get_data_loader(dataset_name, data_path, batch_size, mask_tag, minimum_observation,
                                      reconstruct_input, predict_label)

    train_dataloader = dataloader_dict['train']
    iter_idx = 0

    while iter_idx < 2000:
        for batch in train_dataloader:
            iter_idx += 1
            concat_input, input_feature_list, input_time_list, input_mask_list, label_feature_list, \
                label_time_list, label_mask_list, type_list, input_len_list, label_len_list = batch
            optimizer.zero_grad()

            samp_trajs = torch.stack(input_feature_list, dim=0)
            samp_ts = torch.FloatTensor(np.array(input_time_list[0]))

            # backward in time to infer q(z_0)f
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)

            if loss_func_name == 'MSE':
                z0 = out[:, :latent_dim]
                # forward in time and solve ode for reconstructions
                pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
                pred_x = dec(pred_z)

                loss = mse_loss_func(pred_x, samp_trajs)
                loss.backward()
                optimizer.step()
            else:
                loss = kl_loss_func(out, latent_dim, samp_trajs, samp_ts, func, noise_std, dec, device)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                kl_loss = kl_loss_func(out, latent_dim, samp_trajs, samp_ts, func, noise_std, dec, device)
                loss_meter_kl.update(kl_loss.item())

                z0 = out[:, :latent_dim]
                pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
                pred_x = dec(pred_z)
                mse_loss = mse_loss_func(pred_x, samp_trajs)
                loss_meter_mse.update(mse_loss)

            print('optimize loss type: {}. Iter: {}, running avg elbo: {:.4f}, MSE: {:.4f}'
                  .format(loss_func_name, iter_idx, loss_meter_kl.avg, loss_meter_mse.avg))


def kl_loss_func(out, latent_dim, samp_trajs, samp_ts, func, noise_std, dec, device):
    qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
    epsilon = torch.randn(qz0_mean.size()).to(device)
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

    # forward in time and solve ode for reconstructions
    pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
    pred_x = dec(pred_z)

    # compute loss
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(
        samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1)
    loss = torch.mean(-logpx + analytic_kl, dim=0)
    return loss


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


if __name__ == '__main__':
    main()
