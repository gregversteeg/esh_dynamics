"""Test a variety of samplers against a variety of datasets. Used to generate MMD plots and ESS table for the paper."""
import os
import numpy as np
import pickle
import torch as t
from torch.utils.tensorboard import SummaryWriter
from esh import datasets, utils, samplers
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy.interpolate import interp1d
plt.style.use('seaborn-paper')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


if __name__ == '__main__':
    # Seed for reproducibility
    seed = 4  # adjusted to get non-overlapping lines
    t.manual_seed(seed)
    if t.cuda.is_available():
        device = t.device('cuda')
        t.cuda.manual_seed_all(seed)
    else:
        device = t.device('cpu')
        print('Warning, no CUDA detected.')

    # Logging
    dataset = 'fig1'
    log_root = os.path.join(os.path.expanduser("~"), 'tmp/esh/{}/'.format(dataset))
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    last_run = max([0] + [int(k) for k in os.listdir(log_root) if k.isdigit()])
    log_dir = os.path.join(log_root, '{0:04d}'.format(last_run + 1))  # Each run log gets a new directory
    os.makedirs(log_dir)
    for folder in ['figs', 'code', 'checkpoints']:
        os.mkdir(os.path.join(log_dir, folder))
    os.system('cp *.py {}/code/'.format(log_dir))
    print("Visualize logs using: tensorboard --logdir={0}".format(log_dir))
    writer = SummaryWriter(log_dir)

    sampler_list = [
                     ('ESH-leap', samplers.leap_integrate, {'epsilon': 0.1}),  # time scaled
                     ('MALA 0.1', samplers.hmc_integrate, {'epsilon': 0.1, 'k': 1, 'mh_reject': True}),
                     ('NH', samplers.nh_integrate, {'epsilon': 0.01}),  # Numerical errors above 0.01
                     ('NUTS', samplers.nuts, {}),  # NUTS automatically finds hyper-parameters
                   ]

    n_steps = 50  # number of gradient steps

    save = []
    e_name, e_model = ('2D MOG-prior', datasets.ToyDataset(toy_type='gmm') )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.set_size_inches(8, 8, forward=True)
    ax.set_aspect('equal')
    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)

    for k, (sampler_name, sampler, kwargs) in enumerate(sampler_list):
        x0 = 0.5 * t.tensor([np.sin(k * np.pi / 4), np.cos(k * np.pi / 4)])  # Choose different modes for initialization, so no overlap
        exp_string = '{}_{}_{}'.format(e_name, sampler_name, kwargs)
        print(exp_string)
        energy = e_model.energy

        xs, vs, ts = sampler(energy, x0, n_steps, **kwargs)
        print('length', len(xs))

        colors = ['tab:blue', 'tab:green', 'cyan', 'magenta', 'blue', 'darkgreen', 'cyan', 'lime']
        #import IPython; IPython.embed()
        if k == 0:
            new_ts = t.linspace(ts[0], ts[-1], steps=500)  # Interpolation grid
            xt = interp1d(ts, xs, kind='cubic', assume_sorted=True, axis=0)  # cubic interpolation
            vt = interp1d(ts, vs, kind='cubic', assume_sorted=True, axis=0)
            xs,  vs, ts = xt(new_ts), vt(new_ts), new_ts
            points = xs.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lws = np.linalg.norm(vs, axis=1)
            lws = 5. * np.clip(lws / lws.mean(), 0.01, 1.)
            lc = LineCollection(segments, linewidths=lws, color='blue')
            ax.add_collection(lc)
        else:
            ax.plot(xs[:, 0], xs[:, 1], c=colors[k], lw=3.5)

    r = 1
    n_x = 100
    alpha = 0.7
    xv, yv = np.meshgrid(np.linspace(-r, r, n_x), np.linspace(-r, r, n_x))
    x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
    with t.no_grad():
        energy_all = energy(t.tensor(x_grid, dtype=xs.dtype)).cpu().numpy()

    e_history_grid = energy_all.reshape((n_x, n_x))
    xs_grid = x_grid[:, 0].reshape((n_x, n_x))
    ys_grid = x_grid[:, 1].reshape((n_x, n_x))
    p_grid = np.exp(-e_history_grid) / np.sum(np.exp(-e_history_grid))
    grid = [-4. + 0.1 + i + np.log(p_grid.max()) for i in range(5)]
    ax.contourf(xs_grid, ys_grid, np.log(p_grid), grid,
                # np.exp([-10,  -5,  -2.5,  -1.25, -0.5,  -0.1,   0.2,   0.9, 1.5]),
                cmap="OrRd", zorder=0, alpha=alpha)

    filename = '{}/figs/trajectory_{}.png'.format(log_dir, exp_string)
    fig.savefig(filename, transparent=True)
    plt.close(fig)