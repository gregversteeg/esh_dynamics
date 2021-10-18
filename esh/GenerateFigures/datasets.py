"""
Examples of dataset classes.
The data class just has to have a "sample_data" function.
."""
import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
import matplotlib.pyplot as plt
import scipy.stats as ss


class Funnel():
    """This implementation from the paper "A Neural Network MCMC sampler that maximizes Proposal Entropy"."""
    def __init__(self, d=20, sig=3, clip_y=11):
        self.d = d
        self.sig = sig
        self.clip_y = clip_y
        self.true_mean = 0
        self.true_var = t.tensor([sig**2,] + [(np.exp((sig/2)**2)-1) * np.exp((sig/2)**2)] * (d-1))
        second_scale = t.tensor([12.7,] + [1730.] * (d-1))  # Empirical from a large number of samples
        self.true_scale = t.sqrt(t.cat([t.tensor(self.true_var), second_scale]))

    def energy(self, x):
        if len(x.shape) == 1:
            x = x.view((1, -1))
        E_y = x[:, 0].pow(2) / (2 * self.sig ** 2)  # sig being the variance of first dimension
        E_x = x[:, 1:].pow(2).flatten(1).sum(1) * (x[:, 0].clamp(-25, 25).exp()) / 2 - ((self.d - 1) / 2) * x[:, 0]
        return E_y + E_x

    def sample_data(self, N_samples):
        # sample from Nd funnel distribution
        y = (self.sig * t.randn((N_samples, 1))).clamp(-self.clip_y, self.clip_y)
        x = t.randn((N_samples, self.d - 1)) * (-y / 2).exp()
        return t.cat((y, x), dim=1)

    def inverse(self, y):
        y_inv = t.exp(y[:,0] / 2).view((-1, 1)) * y
        return t.cat((y[:,:1] / self.sig, y_inv[:, 1:]), dim=1)


class Quadratic(nn.Module):
    """Quadratic energy model"""
    def __init__(self, d, dtype=t.float32):
        super().__init__()
        self.dtype = dtype
        self.weight = nn.Parameter(t.zeros((d, d), dtype=dtype))
        self.weight.data.uniform_(-1e-5, 1e-5)
        self.weight.data.fill_diagonal_(1.)

    def sym_weight(self):
        return 0.5 * (self.weight + self.weight.T)

    def forward(self, x):
        return 0.5 * t.einsum('...j,jk,...k->...', x, self.sym_weight(), x)


class GaussianTest:
    """Gaussian samplers and energy models with ground truth moments etc."""
    def __init__(self, d=50, rotate=False, device='cpu', dtype=t.float32, eig=None):
        self.d = d
        if rotate == 'random':
            self.rot = t.tensor(ss.ortho_group.rvs(dim=d), dtype=dtype)
        elif rotate == 'strong':
            assert d == 2, 'only implemented for 2-d'
            self.rot = t.tensor([[np.cos(np.pi / 4), np.sin(np.pi / 4)], [-np.sin(np.pi / 4), np.cos(np.pi / 4)]], dtype=dtype)
        else:
            self.rot = t.eye(d, dtype=dtype)
        self.eig = t.linspace(0.01, 1., d, dtype=dtype)  # Example eigenvalues from Neal 2010 HMC paper
        if rotate == 'strong':
            self.eig = t.tensor([0.01, 1.], dtype=dtype)  # rho = 0.99
        if eig is not None:
            self.eig = t.tensor(eig, dtype=dtype)
        self.cov = t.einsum('ij,j,jk', self.rot.T, self.eig, self.rot)
        self.prec = t.einsum('ij,j,jk', self.rot.T, 1. / self.eig, self.rot)
        self.logdet = t.log(self.eig).sum()
        self.device = device
        self.energy = Quadratic(self.d)
        self.energy.weight.data = self.prec
        self.true_mean = 0.
        self.true_var = t.diagonal(self.cov, 0)
        self.true_scale = t.sqrt(t.cat([t.tensor(self.true_var), 2 * t.tensor(self.true_var ** 2)]))
        # self.scale = t.Tensor(1. / np.sqrt(np.dot(self.rot.T**2, self.eig)))


    def sample_data(self, batch_size):
        device = self.device
        unit_normal = t.randn(batch_size, self.d, device=device)
        return t.einsum('ij,j,jk->ik', unit_normal, t.sqrt(self.eig).to(device), self.rot.to(device))
        # return t.einsum('ij,j,jk,k->ik', unit_normal,
                        # t.sqrt(self.eig).to(device), self.rot.to(device), self.scale.to(device))


    def inverse(self, y):
        """Take data from this Gaussian and tronsform back to standard normal."""
        device = y.device
        return t.einsum('ij,k,kj->ik', y, 1. / t.sqrt(self.eig).to(device), self.rot.to(device))


class ToyDataset:
    """
    Adapted from https://github.com/point0bar1/ebm-anatomy/blob/master/utils.py, Anatomy of MCMC paper.
    This class generates different 2d datasets (rings and gaussian mixtures) along with ground truth and some
    visualizations.
    """
    def __init__(self, toy_type='gmm', toy_groups=8, toy_sd=0.075, toy_radius=0.5, viz_res=500, kde_bw=0.05):
        # import helper functions
        from scipy.stats import gaussian_kde
        from scipy.stats import multivariate_normal
        self.gaussian_kde = gaussian_kde
        self.mvn = multivariate_normal

        # toy dataset parameters
        self.toy_type = toy_type
        self.toy_groups = toy_groups
        self.toy_sd = toy_sd
        self.toy_radius = toy_radius
        self.weights = np.ones(toy_groups) / toy_groups
        if toy_type == 'gmm':
            means_x = np.cos(2*np.pi*np.linspace(0, (toy_groups-1)/toy_groups, toy_groups)).reshape(toy_groups, 1, 1, 1)
            means_y = np.sin(2*np.pi*np.linspace(0, (toy_groups-1)/toy_groups, toy_groups)).reshape(toy_groups, 1, 1, 1)
            self.means = toy_radius * np.concatenate((means_x, means_y), axis=1)
        else:
            self.means = None

        # ground truth density
        if self.toy_type == 'gmm':
            def true_density(x):
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k]*self.mvn.pdf(np.array([x[1], x[0]]), mean=self.means[k].squeeze(),
                                                            cov=(self.toy_sd**2)*np.eye(2))
                return density
            def true_energy_model(x):
                means = t.tensor(self.means.reshape((-1, 1, 2)), device=x.device)
                c = np.log(self.toy_groups * 2 * np.pi * self.toy_sd**2)
                f = -t.logsumexp(t.sum(-0.5 * t.square((x - means) / self.toy_sd), dim=2), dim=0) + c
                return f

            self.true_mean = 0.
            self.true_var = toy_sd ** 2 + np.mean(self.means[:, :, 0, 0] ** 2, axis=0)
            self.true_scale = t.sqrt(t.tensor([0.1305, 0.1308, 0.0107, 0.0107]))  # estimated for toy_groups=8, toy_sd=0.075, toy_radius=0.5 with 100k samples
        elif self.toy_type == 'rings':
            def true_density(x):
                radius = np.sqrt((x[1] ** 2) + (x[0] ** 2))
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k] * self.mvn.pdf(radius, mean=self.toy_radius * (k + 1),
                                                              cov=(self.toy_sd**2))/(2*np.pi*self.toy_radius*(k+1))
                return density
        else:
            raise RuntimeError('Invalid option for toy_type (use "gmm" or "rings")')
        self.true_density = true_density

        self.energy = true_energy_model

        # viz parameters
        self.viz_res = viz_res
        self.kde_bw = kde_bw
        if toy_type == 'rings':
            self.plot_val_max = toy_groups * toy_radius + 4 * toy_sd
        else:
            self.plot_val_max = toy_radius + 4 * toy_sd


    def sample_data(self, num_samples, device='cpu'):
        toy_sample = np.zeros(0).reshape(0, 2, 1, 1)
        sample_group_sz = np.random.multinomial(num_samples, self.weights)
        if self.toy_type == 'gmm':
            for i in range(self.toy_groups):
                sample_group = self.means[i] + self.toy_sd * np.random.randn(2*sample_group_sz[i]).reshape(-1, 2, 1, 1)
                toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
                np.random.shuffle(toy_sample)
        elif self.toy_type == 'rings':
            for i in range(self.toy_groups):
                sample_radii = self.toy_radius*(i+1) + self.toy_sd * np.random.randn(sample_group_sz[i])
                sample_thetas = 2 * np.pi * np.random.random(sample_group_sz[i])
                sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(-1, 1)
                sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(-1, 1)
                sample_group = np.concatenate((sample_x, sample_y), axis=1)
                toy_sample = np.concatenate((toy_sample, sample_group.reshape(-1, 2, 1, 1)), axis=0)
        else:
            raise RuntimeError('Invalid option for toy_type ("gmm" or "rings")')
        return t.Tensor(toy_sample[:,:,0,0]).to(device)

    def plot_toy_density(self, plot_truth=False, f=None, epsilon=0.0, x_s_t=None, save_path='toy.pdf'):
        # save values for plotting groundtruth landscape
        self.xy_plot = np.linspace(-self.plot_val_max, self.plot_val_max, self.viz_res)
        self.z_true_density = np.zeros(self.viz_res**2).reshape(self.viz_res, self.viz_res)
        for x_ind in range(len(self.xy_plot)):
            for y_ind in range(len(self.xy_plot)):
                self.z_true_density[x_ind, y_ind] = self.true_density([self.xy_plot[x_ind], self.xy_plot[y_ind]])

        num_plots = 0
        if plot_truth:
            num_plots += 1

        # density of learned EBM
        if f is not None:
            num_plots += 1
            xy_plot_torch = t.Tensor(self.xy_plot).view(-1, 1, 1, 1).to(next(f.parameters()).device)
            # y values for learned energy landscape of descriptor network
            z_learned_energy = np.zeros([self.viz_res, self.viz_res])
            for i in range(len(self.xy_plot)):
                y_vals = float(self.xy_plot[i]) * t.ones_like(xy_plot_torch)
                vals = t.cat((xy_plot_torch, y_vals), 1)
                z_learned_energy[i] = f(vals[:,:,0,0]).data.cpu().numpy()
            # rescale y values to correspond to the groundtruth temperature

            # transform learned energy into learned density
            z_learned_density_unnormalized = np.exp(- (z_learned_energy - np.min(z_learned_energy)))
            bin_area = (self.xy_plot[1] - self.xy_plot[0]) ** 2
            z_learned_density = z_learned_density_unnormalized / (bin_area * np.sum(z_learned_density_unnormalized))

        # kernel density estimate of shortrun samples
        if x_s_t is not None:
            num_plots += 1
            density_estimate = self.gaussian_kde(x_s_t.squeeze().cpu().numpy().transpose(), bw_method=self.kde_bw)
            z_kde_density = np.zeros([self.viz_res, self.viz_res])
            for i in range(len(self.xy_plot)):
                for j in range(len(self.xy_plot)):
                    z_kde_density[i, j] = density_estimate((self.xy_plot[j], self.xy_plot[i]))

        # plot results
        plot_ind = 0
        fig = plt.figure()

        # true density
        if plot_truth:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('True density')
            plt.imshow(self.z_true_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('True log-density')
            plt.imshow(np.log(self.z_true_density + 1e-10), cmap='viridis')
            plt.axis('off')
        # learned ebm
        if f is not None:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('EBM density')
            plt.imshow(z_learned_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('EBM log-density')
            plt.imshow(np.log(z_learned_density + 1e-10), cmap='viridis')
            plt.axis('off')
        # shortrun kde
        if x_s_t is not None:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('Short-run KDE')
            plt.imshow(z_kde_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('Short-run log-KDE')
            plt.imshow(np.log(z_kde_density + 1e-10), cmap='viridis')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.close()
