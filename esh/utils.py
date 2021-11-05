"""Library of methods for visualization with a few miscellaneous methods."""
import numpy as np
import torch as t
import torchvision as tv
from esh import datasets

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections as mc
from matplotlib.collections import LineCollection
import seaborn as sns
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.style.use('seaborn-paper')  # also try 'seaborn-talk', 'fivethirtyeight'
try:
    plt.style.use('/Users/gregv/Dropbox/Public/gv3.mplstyle')
except:
    pass

##########################################################
# Plots
##########################################################
def viz_trajectory(energy, xs, trajectory, filename):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.set_size_inches(8, 8, forward=True)
    ax.set_aspect('equal')
    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)

    n_x = int(np.sqrt(energy.shape[0]))  # number of x-grid locations to sample energy for contour plot
    e_history_grid = energy.reshape((n_x, n_x))
    xs_grid = xs[:, 0].reshape((n_x, n_x))
    ys_grid = xs[:, 1].reshape((n_x, n_x))
    cont = plt.contourf(xs_grid, ys_grid, np.exp(-e_history_grid), [1e-13, 0.01, 0.1, 0.3, 0.6, 0.9, 1.2, 2.4, 4.8], cmap="OrRd", zorder=0)

    # plt.colorbar()
    ax.plot(trajectory[:,0], trajectory[:,1], color='black', linewidth=1, alpha=1.)
    k = max(len(trajectory) // 100, 1)
    ax.scatter(trajectory[::k,0], trajectory[::k,1], alpha=0.3)

    fig.savefig(filename, transparent=True)
    plt.close(fig)


def viz_weighted_trajectory(xs, ts, f=None, weights=False, n_x=100, alpha=0.7, filename='weighted_trajectory.png'):
    """Trajectory with line width proportional to time. f can be an energy model contour plot
    for the background, n_x is resolution."""
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.set_size_inches(8, 8, forward=True)
    ax.set_aspect('equal')
    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)

    if f is not None:
        if xs.abs().max() > 1:
            r = 3
        else:
            r = 1
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

    if weights:
        plt.plot(xs[:,0], xs[:,1], lw=0.01)
        points = xs.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lengths = t.square(xs[1:] - xs[:-1]).sum(axis=1).sqrt()
        lws = (ts[1:] - ts[:-1]) / lengths
        lws = 10. * t.clamp(lws / lws.mean(), 0.05, 1.)

        lc = LineCollection(segments, linewidths=lws, color='blue')
        ax.add_collection(lc)
    else:
        colors = ['tab:blue', 'tab:green', 'tab:brown', 'tab:purple', 'blue', 'darkgreen', 'cyan', 'lime'] * 10
        plt.plot(xs[:,0], xs[:,1], c=colors[viz_weighted_trajectory.count])

    fig.savefig(filename, transparent=True)
    plt.close(fig)

viz_weighted_trajectory.count = 0


def trajectory_movie(xs, f=None, output='figs/trajectory.mp4', decay=0.85, n_x=100, alpha=0.7):
    """t is the time series of trajectories. loop, time, (x,y) location."""
    assert xs.shape[2] == 2, "Dimensions should be loop, time, (x,y) location"
    ns = xs.shape[1] - 1
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.set_size_inches(8, 8, forward=True)
    ax.set_aspect('equal')
    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)

    if f is not None:
        if np.max(np.abs(xs)) > 1:
            r = 3
        else:
            r = 1
        xv, yv = np.meshgrid(np.linspace(-r, r, n_x), np.linspace(-r, r, n_x))
        x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
        with t.no_grad():
            energy = f(t.tensor(x_grid)).cpu().numpy()

        e_history_grid = energy.reshape((n_x, n_x))
        xs_grid = x_grid[:, 0].reshape((n_x, n_x))
        ys_grid = x_grid[:, 1].reshape((n_x, n_x))
        p_grid = np.exp(-e_history_grid) / np.sum(np.exp(-e_history_grid))
        grid = [-4. + 0.1 + i + np.log(p_grid.max()) for i in range(5)]
        ax.contourf(xs_grid, ys_grid, np.log(p_grid), grid,  # np.exp([-10,  -5,  -2.5,  -1.25, -0.5,  -0.1,   0.2,   0.9, 1.5]),
                    cmap="OrRd", zorder=0, alpha=alpha)

    scat = ax.scatter(xs[:, 0, 0], xs[:, 0, 1], s=60, zorder=10)

    cols = []
    for ti in xs:
        ls = np.array([ti[:-1], ti[1:]]).transpose((1, 0, 2))
        lc = mc.LineCollection(ls, linewidths=2, colors=(0,0,0,0))
        col = ax.add_collection(lc)
        cols.append(col)

    ax.set_title("t={}".format(0))
    ax.set_xlim(left=np.min(xs[:,:,0]), right=np.max(xs[:,:,0]))
    ax.set_ylim(bottom=np.min(xs[:,:,1]), top=np.max(xs[:,:,1]))
    ax.set_xlabel('Location ($x_1$)')
    ax.set_ylabel('Location ($x_2$)')

    def update_plot(i):
        for col in cols:
            col.set_color([(0,0,0, decay**(i-j)) if j < i else (0,0,0,0) for j in range(ns)])
        scat.set_offsets(xs[:, i])
        ax.set_title("t = {}".format(i), fontsize=24)

    ani = animation.FuncAnimation(fig, update_plot, frames=range(ns))
    ani.save(output, writer=animation.FFMpegWriter(fps=20))
    plt.close(fig)


def esh_diagnostics(xs, vs, ts, f, log_dir, exp_string, original_var=None):
    if len(xs.shape) == 2:
        d = xs.shape[1]
        xs = xs.view((-1, 1, d))
        vs = vs.view((-1, 1, d))
    elif len(xs.shape) == 4:
        vs = vs.flatten(start_dim=2)
    for i in range(xs.shape[1]):
        K_t = t.log(t.square(vs[:,i]).mean(axis=1))
        plt.plot(K_t)
    plt.xlabel('time')
    plt.ylabel('$d/2 \log v^2 / d$')
    plt.tight_layout()
    plt.savefig(log_dir + '/figs/log_v2_t_{}.png'.format(exp_string), transparent=True)
    plt.clf()

    for i in range(xs.shape[1]):
        K_t = t.log(t.square(vs[:,i]).mean(axis=1))
        H_t = K_t + f(xs[:,i])
        plt.plot(H_t)
    plt.xlabel('time')
    plt.ylabel('$H(x,v)$')
    plt.tight_layout()
    plt.savefig(log_dir + '/figs/H_t_{}.png'.format(exp_string), transparent=True)
    plt.clf()

    #import IPython; IPython.embed()
    if original_var is not None:
        xs, vs, ts = original_var
        K_t = t.square(vs).mean(axis=1)
        dts = ts[1:] - ts[:-1]
        summary = 'Number of accepted steps {} / {}. Total time: {}'.format((dts > 0.).sum(), len(dts), ts[-1])
        plt.scatter(dts[dts > 0.], K_t[:-1][dts > 0.])
        plt.xlabel('$\Delta t$, step size')
        plt.ylabel('$v^2 / d$')
        plt.title(summary)
        plt.tight_layout()
        plt.savefig(log_dir + '/figs/steps_{}.png'.format(exp_string), transparent=True)
        plt.clf()


def esh_diagnostics_2(xs, rs, x_shape, f, log_dir, exp_string):
    """ Size of xs is (time, batch, d) and for r is time, batch, 1. """
    d = np.prod(xs.shape[2:])
    for i in range(xs.shape[1]):
        K_t = d * rs[:,i]
        plt.plot(K_t.cpu())
    plt.xlabel('time')
    plt.ylabel('$d/2 \log v^2 / d$')
    plt.tight_layout()
    plt.savefig(log_dir + '/figs/log_v2_t_{}.png'.format(exp_string), transparent=True)
    plt.clf()

    with t.no_grad():
        for i in range(xs.shape[1]):
            K_t = d * rs[:,i]
            H_t = K_t + f(xs[:,i].view((-1,) + x_shape))
            plt.plot(H_t.cpu())
    plt.xlabel('time')
    plt.ylabel('$H(x,v)$')
    plt.tight_layout()
    plt.savefig(log_dir + '/figs/H_t_{}.png'.format(exp_string), transparent=True)
    plt.clf()


def plot_all_mmd(save, log_dir):
    sampler = {'2D MOG': datasets.ToyDataset(toy_type='gmm'),
               '2D MOG-prior': datasets.ToyDataset(toy_type='gmm'),
               '50D ICG': datasets.GaussianTest(50, rotate=False),
               '2D SCG': datasets.GaussianTest(2, rotate='strong'),
               '2D SCG-bias': datasets.GaussianTest(2, rotate='strong'),
               '20D Funnel': datasets.Funnel()}

    datasets_in_run = set([run[0] for run in save])
    for sampler_key in sampler:
        if sampler_key in datasets_in_run:
            data_model = sampler[sampler_key]
            y = data_model.sample_data(10000)
            if hasattr(data_model, 'inverse'):
                inv = data_model.inverse
                y = inv(y)
            else:
                inv = lambda q: q
            bw = median_heuristic(y)  # is pretty bad for funnel.
            print('Sampler bandwidth:', sampler_key, bw)
            kyy = kernel(y, y, bw)

            x_plot_offset = 0.  # avoid overlapping error bars
            fig1, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
            df = pd.DataFrame(columns=['Method', 'Grad evals', 'x', 'y'])  # dataframe used by seaborn to make a grid plot
            for run in save:
                e_name, sampler_name, exp_string, n_repeat, n_steps, all_xs, all_vs, all_ts = run
                if e_name == sampler_key:
                    print('MMD processing', e_name, sampler_name)
                    new_x, new_t = grad_step_fill(all_xs, all_ts, n_steps, ergodic = (sampler_name[:4] == 'ESH-'))
                    mmds = []
                    cis = []
                    ts = []
                    for i in range(0, n_steps + 1, 10):
                        print(i)
                        x = inv(new_x[i])
                        dfi = pd.DataFrame(new_x[i, :200, [0, -1]].cpu().numpy(), columns=['x', 'y'])
                        dfi['Method'] = sampler_name
                        dfi['Grad evals'] = new_t[i].cpu().numpy()
                        df = df.append(dfi)
                        mmd_mean, mmd_ci = mmd_ci_kernel(kernel(x, x, bw), kernel(x, y, bw), kyy)
                        mmds.append(mmd_mean)
                        cis.append(mmd_ci)
                        ts.append(i + x_plot_offset)
                    cis = (t.stack(cis).T - t.tensor(mmds)).abs()
                    ax1.errorbar(ts, mmds, yerr=cis.numpy(), label=sampler_name)
                    x_plot_offset += 0.3

            ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")
            # plt.subplots_adjust(right=0.7)
            ax1.set_xlabel('# gradient evaluations')
            ax1.set_ylabel('MMD')
            fig1.savefig(log_dir + '/figs/mmd_{}.png'.format(sampler_key), transparent=True, bbox_inches="tight")

            with sns.axes_style("white"):
                g = sns.FacetGrid(df[df['Grad evals'] <= 40], row="Method", col="Grad evals", margin_titles=True)
                g.map(sns.regplot, "x", "y", color=".3", fit_reg=False)
            g.fig.subplots_adjust(wspace=.02, hspace=.02)
            g.set_axis_labels("$x_1$", "$x_2$")
            g.savefig(log_dir + '/figs/sample_evolution_{}.png'.format(sampler_key), transparent=True, bbox_inches="tight")


def plot_all_ess(save, log_dir):
    e_models = {'2D MOG': datasets.ToyDataset(toy_type='gmm'),
                '2D MOG-prior': datasets.ToyDataset(toy_type='gmm'),
               '50D ICG': datasets.GaussianTest(50, rotate=False),
               '2D SCG': datasets.GaussianTest(2, rotate='strong'),
                '2D SCG-bias': datasets.GaussianTest(2, rotate='strong'),
                '20D Funnel': datasets.Funnel()}

    all_results = []
    for run in save:
        e_name, sampler_name, exp_string, n_repeat, n_steps, all_xs, all_vs, all_ts = run

        e_model = e_models[e_name]
        # If ground truth is known, record scaled moments for ESS and MSE
        if hasattr(e_model, 'true_mean') and hasattr(e_model, 'true_var') and hasattr(e_model, 'true_scale'):
            esss = []
            for xs in all_xs:
                moments = t.cat([xs - e_model.true_mean, t.square(xs) - t.tensor(e_model.true_var)], 1) / e_model.true_scale
                esss.append(ESS(moments).min().item() / n_steps)
        all_results.append((e_name, sampler_name, np.mean(esss), np.std(esss)))
    df = pd.DataFrame(all_results, columns = ['Dataset', 'Sampler', 'Mean', 'Std'])

    df['mean (std)'] = ['{:.2e} ({:.1e})'.format(m, s) for m, s in zip(df.Mean, df.Std)]
    print(df.pivot(values='mean (std)', index='Dataset', columns='Sampler'))
    f = open(log_dir + '/ess.txt', 'w')
    f.write(df.pivot(values='mean (std)', index='Dataset', columns='Sampler').to_latex())
    f.close()


def image_movie(im_history, filename):
    n_t = len(im_history)  # number of time-steps
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.set_title("t = {}".format(0))
    ax.set_aspect('equal')
    plt.axis('off')
    nrows = int(np.sqrt(len(im_history[0])))

    im_history = np.clip(im_history*0.5 + 0.5, 0, 1)
    get_grid = lambda x: tv.utils.make_grid(t.Tensor(x), nrow=nrows).cpu().numpy().transpose((1, 2, 0))
    im = ax.imshow(get_grid(im_history[0]))

    # animation function
    def animate(i):
        im.set_array(get_grid(t.Tensor(im_history[i])))
        plt.title('t = %i' % i)

    anim = animation.FuncAnimation(fig, animate, frames=n_t)
    anim.save(filename, writer=animation.FFMpegWriter(fps=10))
    plt.close(fig)
##########################################################
# Metrics
##########################################################
def autocovariance(xs, tau=0):
    return (xs[:len(xs)-tau] * xs[tau:]).mean(dim=0)


def ESS(X, scale=1):
    """From Hoffman & Gelman, 2014."""
    total_t = X.shape[0]
    A = t.stack([autocovariance(X / scale, tau=t) for t in range(total_t-1)])

    one_minus_s_over_M = (1 - t.arange(len(A)) / len(A)).view((-1, 1))
    A = A * t.cumprod((A>0.05), axis=0) * one_minus_s_over_M
    return total_t / (1. + 2 * t.sum(A[1:], axis=0))


def kernel(x, y, bw):
    """Gaussian kernel with adjustable bandwidth."""
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    dist = t.square(x - y.T).sum(axis=1)
    return t.exp(- 0.5 * dist / bw) / t.sqrt(t.tensor(2. * np.pi * bw))


def median_heuristic(x):
    """Gretton et al heuristic, use the median of all distances as the bandwidth"""
    x = x.view(x.shape[0], x.shape[1], 1)
    return t.square(x - x.transpose(0, 2)).sum(axis=1).median()


def mmd_ci_kernel(kxx, kxy, kyy):
    """Maximum mean discrepancy from the kernels. It's faster not to re-compute some kernels each time for my plots.
    This is a special version where we also estimate the bootstrap conf. interval of the estimator for each x_i"""
    iuy = t.triu_indices(len(kyy), len(kyy), 1)
    cost = t.mean(kyy[iuy[0], iuy[1]])
    cost = cost + (t.sum(kxx, axis=1) - kxx.diag()) / (len(kxx) - 1)
    cost = cost - 2 * t.mean(kxy, axis=1)
    return cost.mean(), bootstrap_ci(cost)


def bootstrap_ci(x, num_samples=1000):
    """Give a 95% bootstrap CI for the mean of a 1-d pytorch tensor, using 1000 bootstrap samples"""
    n = len(x)
    bootstrap_samples = x[t.randint(0, n, (num_samples, n))]  # Sample with replacement, 1000 times
    means = bootstrap_samples.mean(dim=1)  # 1000 means
    interval = means.sort().values[[int(0.025 * num_samples), -int(0.025 * num_samples)]]
    return interval


def maximum_mean_discrepancy(x, y):
    """Maximum mean discrepancy (Gretton et al 2012) using Gaussian kernel with adjustable bandwidth.
    Am I crazy? Every single code implementation I found online forgot to exclude the diagonals. Is there some reason
    not to? It seems like it adds some unnecessary bias. I mean, probably not a big deal but still..."""
    iux = t.triu_indices(len(x), len(x), 1)
    iuy = t.triu_indices(len(y), len(y), 1)
    cost = t.mean(kernel(x, x)[iux[0], iux[1]])
    cost += t.mean(kernel(y, y)[iuy[0], iuy[1]])
    cost -= 2 * t.mean(kernel(x, y))
    return cost


##########################################################
# Miscellaneous
##########################################################
def lin_interp(xs, ts, multiplier=1.):
    """Linearly interpolate so that the ts fall on a regular grid,
    and xs are linearly interpolated to fall on that grid.  Multiplier changes the number of points on the grid,
    compared to the original grid."""
    n = int(multiplier * len(ts))  # total number of new time steps
    grid_ts = t.linspace(ts[0], ts[-1], n, dtype=ts.dtype)
    grid_xs = t.zeros((n,) + xs.shape[1:], device=xs.device, dtype=xs.dtype)
    for j, tt in enumerate(grid_ts):
        if len(t.nonzero(tt > ts)):  # the grid time point is greater than some sampled point
            i = t.nonzero(tt > ts)[-1][0].item()  # find the smallest sampled point it is greater than
        else:
            i = t.nonzero(ts)[0][0] - 1  # the last value where t=0
        if i > n - 2:  # Shouldn't be needed, but sometimes the equal floats were considered unequal
            i = n - 2
        if np.abs(tt - ts[i]) < 1e-8:
            grid_xs[j] = xs[i]
        else:
            grid_xs[j] = xs[i] + (xs[i+1] - xs[i]) / (ts[i+1] - ts[i]) * (tt - ts[i])
    return grid_xs, grid_ts


def grad_step_fill(xs, ts, n_steps, ergodic=False):
    """For a list of xs, return an array that has one time-step for each number of gradients.
    Ergodic sampling takes a random point from each time-series, up until a given number of gradients in the chain."""
    shape = xs[0][0].shape

    batch_size = len(xs)
    new_x = t.zeros((n_steps + 1, batch_size,) + shape)
    new_t = t.arange(n_steps + 1)

    for i, (x, tt) in enumerate(zip(xs, ts)):
        t_grads = t.linspace(0., n_steps, len(x))  # Assumes gradients are equally spaced, true for samplers in "samplers.py"
        for j in range(n_steps + 1):
            if len(t.nonzero(t_grads <= j)):
                k = t.nonzero(t_grads <= j)[-1][0].item()
            else:
                k = 0
            if ergodic:
                random_t = np.random.random() * tt[k]
                right_ind = t.nonzero(random_t <= tt)[0].item()
                if right_ind == 0 or tt[right_ind] == random_t:
                    new_x[j, i] = x[right_ind]
                else:
                    dt = tt[right_ind] - tt[right_ind - 1]
                    x_interp = x[right_ind - 1] + (x[right_ind] - x[right_ind - 1]) * (random_t - tt[right_ind - 1]) / dt
                    new_x[j, i] = x_interp
            else:
                new_x[j, i] = x[k]
    return new_x, new_t
