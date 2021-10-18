"""Test a variety of samplers against a variety of datasets. Used to generate MMD plots and ESS table for the paper."""
import os
import numpy as np
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
    seed = 1  # Seed for reproducibility
    t.manual_seed(seed)
    if t.cuda.is_available():
        device = t.device('cuda')
        t.cuda.manual_seed_all(seed)
    else:
        device = t.device('cpu')
        print('Warning, no CUDA detected.')

    # Logging
    dataset = 'ergodic'
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


    save = []
    e_name, e_model = ('2D MOG-prior', datasets.ToyDataset(toy_type='gmm') )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.set_size_inches(8, 8, forward=True)
    ax.set_aspect('equal')
    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)
    sampler_name = 'ESH-leap'
    sampler = samplers.leap_integrate
    kwargs = {'epsilon': 0.001}  # Small step size because we want accurate dynamics - noise due to large step size may actually *improve* mixing
    n_steps = 500000  # number of gradient steps

    x0 = 0.5 * t.tensor([0., 1.])  # Choose different modes for initialization, so no overlap
    exp_string = 'ergodic'
    print(exp_string)
    energy = e_model.energy

    xs, vs, ts = sampler(energy, x0, n_steps, **kwargs)  # ESH sampler

    ax.plot(xs[:, 0], xs[:, 1], c='tab:blue', lw=1.5, alpha=0.8, zorder=1)  # Plot trajectories

    # Get samples from trajectory
    weights = vs.norm(dim=1).numpy()
    inds = np.random.choice(len(weights), size=500, replace=True, p=weights / sum(weights))
    x = xs[inds]

    ax.scatter(x[:,0], x[:,1], c='tab:green', zorder=2, alpha=1)  # Plot samples

    # Compute MMD
    y = e_model.sample_data(10000)
    bw = utils.median_heuristic(y)
    kyy = utils.kernel(y, y, bw)
    mmd_mean, mmd_ci = utils.mmd_ci_kernel(utils.kernel(x, x, bw), utils.kernel(x, y, bw), kyy)

    ax.set_title("MMD = {:.5f} , 95% CI = [{:.5f},{:.5f}]".format(mmd_mean, mmd_mean+mmd_ci[0], mmd_mean+mmd_ci[1]))

    # Show energy model
    r = 1
    n_x = 100
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
                cmap="OrRd", zorder=0, alpha=0.7)

    # save the plot
    filename = '{}/figs/trajectory_{}.png'.format(log_dir, exp_string)
    fig.savefig(filename, transparent=True)
    plt.close(fig)

