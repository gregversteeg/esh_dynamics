"""Produce a plot for the paper, comparing ESH MC steps to Langevin dynamics."""
import numpy as np
import torch as t
from esh import samplers, datasets
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-paper')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


if __name__ == '__main__':
    # Seed for reproducibility
    scenario = 1  # 0 for deep well 1 for valley
    name = ['deepwell', 'valley'][scenario]
    print('scenario: {}'.format(name))
    seed = 0
    t.manual_seed(seed)
    n_steps = [5, 50][scenario]  # 5 or 50

    m = datasets.GaussianTest(2)
    m.energy.weight.data[0, 0] = [100,1][scenario]  # 1 for valley, 100 for deep well
    m.energy.weight.data[1, 1] = 100. 
    f = m.energy
    r, n_x, alpha = 3, 100, 0.7
    xv, yv = np.meshgrid(np.linspace(-r, r, n_x), np.linspace(-r, r, n_x))
    x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
    with t.no_grad():
        energy = f(t.tensor(x_grid, dtype=t.float32)).cpu().numpy()

    e_history_grid = energy.reshape((n_x, n_x))
    xs_grid = x_grid[:, 0].reshape((n_x, n_x))
    ys_grid = x_grid[:, 1].reshape((n_x, n_x))
    p_grid = np.exp(-e_history_grid) / np.sum(np.exp(-e_history_grid))
    grid = [-4. + 0.1 + i + np.log(p_grid.max()) for i in range(5)]

    xs, vs, ts = samplers.hmc_integrate(f, t.tensor([4., -0.4]), n_steps, epsilon=0.1, k=1, mh_reject=False)
    nxs, nvs = samplers.newton_dynamics(f, t.tensor([4., 0.4]), n_steps, epsilon=0.1)
    exs, evs, ets = samplers.leap_integrate(f, t.tensor([-4., 0.4]), n_steps, epsilon=0.1)

    # import IPython; IPython.embed()
    # %matplotlib
    fig, ax = plt.subplots(1,1)
    ax.contourf(xs_grid, ys_grid, np.log(p_grid), grid,
                # np.exp([-10,  -5,  -2.5,  -1.25, -0.5,  -0.1,   0.2,   0.9, 1.5]),
                cmap="OrRd", zorder=0, alpha=alpha)
    ax.axis('off')
    ax.plot(xs[:, 0], xs[:, 1], label='Langevin dynamics', color='tab:orange')
    ax.text(1., -0.7, 'Langevin dynamics', color='tab:orange')
    ax.plot(nxs[:, 0], nxs[:, 1], label='Newtonian Hamiltonian dynamics', color='tab:green', alpha=0.5)
    ax.text(1., 0.7, 'Newtonian dynamics', color='tab:green')
    ax.plot(exs[:, 0], exs[:, 1], label='ESH dynamics', color='tab:blue')
    ax.text(-4, 0.7, 'ESH dynamics', color='tab:blue')
    fig.savefig('figs/{}.pdf'.format(name), bbox_inches='tight')
