"""
Train a neural net energy model on a 2d toy dataset and visualize the process.
"""

import os
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from esh import datasets
import esh
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
plt.style.use('seaborn-paper')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


class EnergySmallFC(nn.Module):
    """Energy model for 2d input data.
       Current convention is to output ENERGY.
    """
    def __init__(self, w=64, l=0.05, d=2):
        super(EnergySmallFC, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(d, w),
            nn.LeakyReLU(l),
            nn.Linear(w, w),
            nn.LeakyReLU(l),
            nn.Linear(w, w),
            nn.LeakyReLU(l),
            nn.Linear(w, 1))

    def forward(self, x):
        return self.f(x).squeeze()


def viz_energy_sample_movie(e_history, x_history, xs, weight_history):
    # import IPython; IPython.embed()
    n_t = len(x_history)  # number of time-steps
    ns = x_history.shape[1]  # number of trajectories in batch

    # import IPython; IPython.embed()
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.set_size_inches(8, 8, forward=True)
    ax.set_title("t = {}".format(0), fontsize=28)
    ax.set_aspect('equal')
    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)
    n_x = int(np.sqrt(e_history.shape[1]))  # number of x-grid locations to sample energy for contour plot
    e_history_grid = e_history[0].reshape((n_x, n_x))
    xs_grid = xs[:, 0].reshape((n_x, n_x))
    ys_grid = xs[:, 1].reshape((n_x, n_x))
    Z = np.sum(np.exp(-e_history_grid)) * xs.ptp()**2 / n_x**2  # Normalize and density
    levels = [-3,-2,-1,0,1,2, 4]
    cont = [ax.contourf(xs_grid, ys_grid, e_history_grid + np.log(Z), levels, cmap="OrRd", zorder=0)]
    init_size = 30 * weight_history[0].mean() * len(weight_history)
    scat = ax.scatter(x_history[0, :, 0], x_history[0, :, 1], s=init_size, zorder=10)

    def init():
        return [ax]

    # animation function
    def animate(i, decay=0.85):
        scat.set_offsets(x_history[i])
        w = weight_history[i]
        w *= 30 * len(w)
        scat.set_sizes(w)

        for coll in cont[0].collections:
            coll.remove()
        e_history_grid = e_history[i].reshape((n_x, n_x))
        Z = np.sum(np.exp(-e_history_grid)) * xs.ptp() ** 2 / n_x ** 2  # Normalize and density
        cont[0] = ax.contourf(xs_grid, ys_grid, e_history_grid + np.log(Z), levels, cmap="OrRd", zorder=0)
        ax.set_title('t = %i' % i, fontsize=28)
        return cont

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_t)
    anim.save('figs/toy_movie_reservoir.mp4', writer=animation.FFMpegWriter(fps=20))
    plt.close(fig)


if __name__ == '__main__':
    # Seed for reproducibility and set device
    seed = 1
    t.manual_seed(seed)
    if t.cuda.is_available():
        device = t.device('cuda')
        t.cuda.manual_seed_all(seed)
    else:
        device = t.device('cpu')
        print('Warning, no CUDA detected.')

    # Logging
    log_root = os.path.expanduser("~/tmp/esh/train_toy/")
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    last_run = max([0] + [int(k) for k in os.listdir(log_root) if k.isdigit()])
    log_dir = os.path.join(log_root, '{0:04d}'.format(last_run + 1))  # Each run log gets a new directory
    os.makedirs(log_dir)
    for folder in ['figs', 'energy', 'code', 'checkpoints']:
        os.mkdir(os.path.join(log_dir, folder))
    os.system('cp *.py {}/code/'.format(log_dir))
    print("Visualize logs using: tensorboard --logdir={0}".format(log_dir))
    writer = SummaryWriter(log_dir)

    n_iter = 10000
    batch_size = 200
    lr = 0.01
    epsilon = 0.1
    steps = 20
    x_max = 3.

    data_class = datasets.ToyDataset(toy_type='gmm')
    f = EnergySmallFC().to(device)
    optim = t.optim.SGD(f.parameters(), lr=lr)  # Optimizer

    # Store for viz
    n_x = 500  # Resolution of grid
    K = 50  # Sub-sample time-steps
    xv, yv = np.meshgrid(np.linspace(-x_max, x_max, n_x), np.linspace(-x_max, x_max, n_x))
    xs = np.array([xv, yv]).reshape((2, n_x * n_x)).T
    e_history = np.zeros((n_iter // K, n_x * n_x))
    x_history = np.zeros((n_iter // K, batch_size, 2))
    weight_history = np.zeros((n_iter // K, batch_size))

    for i in range(n_iter):
        x_p_d = data_class.sample_data(batch_size, device=device)
        x = 2 * t.rand(batch_size, 2, device=device) - 1
        # x_q, _, _, weights, _, r_s_t = esh_leap.jarzynski_sample(f, x, steps, epsilon)
        _,_,_, x_q, _ = esh.leap_integrate_chain(f, x, steps, epsilon, store=False, reservoir=True)
        weights = t.ones(len(x)) / len(x)

        E_data = f(x_p_d).mean()
        E_samples = t.dot(f(x_q), weights)
        L = E_data - E_samples  # ********************************* Loss function

        optim.zero_grad()
        L.backward()
        optim.step()

        # Logging
        writer.add_scalar('Energy/data', E_data, i)
        writer.add_scalar('Energy/negative', E_samples, i)
        writer.add_scalar('Training loss', L, i)
        writer.add_scalar('Max weight', weights.max(), i)
        writer.add_scalar('Weight variance', weights.var(), i)
        if i % K == 0:
            with t.no_grad():
                e_history[i // K] = f(t.Tensor(xs, device=device)).cpu().numpy()
            x_history[i // K] = x_q.cpu().numpy()
            weight_history[i // K] = weights.cpu().numpy()

    viz_energy_sample_movie(e_history, x_history, xs, weight_history)
    import IPython; IPython.embed()