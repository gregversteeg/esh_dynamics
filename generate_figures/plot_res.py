import numpy as np
import torch as t
from esh import datasets
import esh
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-paper')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

model = datasets.GaussianTest(2, rotate='strong')
energy = model.energy

n_steps = 40
epsilon = 0.1
e_scale = 10.
x,v,r, xr, ur, xr_hist = esh.leap_integrate_chain(energy, t.randn(1, 2), n_steps, epsilon, store=True, reservoir=True, energy_scale=e_scale)

samples = esh.reservoir_binary_search(model.energy, t.randn(100, 2), n_steps, epsilon, 3, energy_scale=e_scale)
%matplotlib

fig, ax = plt.subplots(1,1)
r=3
n_x=100
alpha = 0.7

xv, yv = np.meshgrid(np.linspace(-r, r, n_x), np.linspace(-r, r, n_x))
x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
with t.no_grad():
    energy = model.energy(t.tensor(x_grid, dtype=t.float32)).cpu().numpy()
e_history_grid = energy.reshape((n_x, n_x))
xs_grid = x_grid[:, 0].reshape((n_x, n_x))
ys_grid = x_grid[:, 1].reshape((n_x, n_x))
p_grid = np.exp(-e_history_grid) / np.sum(np.exp(-e_history_grid))
grid = [-4. + 0.1 + i + np.log(p_grid.max()) for i in range(5)]
ax.contourf(xs_grid, ys_grid, np.log(p_grid), grid,
            # np.exp([-10,  -5,  -2.5,  -1.25, -0.5,  -0.1,   0.2,   0.9, 1.5]),
            cmap="OrRd", zorder=0, alpha=alpha)


ax.plot(x[:,0,0],x[:,0,1])
ax.scatter(xr_hist[:,0,0], xr_hist[:,0,1])

ax.scatter(samples[:,0], samples[:,1])


x0 = t.randn(100,2)
for i in range(5):
    samples = esh.reservoir_binary_search(model.energy, x0, n_steps, epsilon, i, energy_scale=e_scale)
    with t.no_grad():
        print(i, model.energy(samples).mean())