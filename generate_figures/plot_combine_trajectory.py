"""
Specify a results file from test_suite.py and then re-plot results.
."""
import os
import pickle
import torch as t
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.style.use('seaborn-paper')  # also try 'seaborn-talk', 'fivethirtyeight'
try:
    plt.style.use(os.path.expanduser('~/Dropbox/Public/gv3.mplstyle'))
except:
    pass

# Put your own test_suite.py run results path in here.
runs = pickle.load(open(os.path.expanduser('~/tmp/neq/sample_all/may 18 ablation re-do with leap fix/results.pkl'), 'rb'))
for run in runs:
    print(run[2])

r=1
n_x=100
alpha=0.7
weights=False
from esh import datasets

f = datasets.ToyDataset(toy_type='gmm').energy

xs = runs[6][-3][0]
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
fig.set_size_inches(8, 8, forward=True)
ax.set_aspect('equal')
plt.axis('off')
fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)

xv, yv = np.meshgrid(np.linspace(-r, r, n_x), np.linspace(-r, r, n_x))
x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
with t.no_grad():
    energy = f(t.tensor(x_grid, dtype=xs.dtype)).cpu().numpy()

e_history_grid = energy.reshape((n_x, n_x))
xs_grid = x_grid[:, 0].reshape((n_x, n_x))
ys_grid = x_grid[:, 1].reshape((n_x, n_x))
p_grid = np.exp(-e_history_grid) / np.sum(np.exp(-e_history_grid))
grid = [-4. + 0.1 + i + np.log(p_grid.max()) for i in range(5)]
ax.contourf(xs_grid, ys_grid, np.log(p_grid), grid,  # np.exp([-10,  -5,  -2.5,  -1.25, -0.5,  -0.1,   0.2,   0.9, 1.5]),
            cmap="OrRd", zorder=0, alpha=alpha)

for i in range(6):
    print(i, runs[6+i][2])
    xs = runs[6+i][-3][0]
    plt.plot(xs[:,0], xs[:,1])

fig.savefig('figs/combine_mog_prior_esh.png')