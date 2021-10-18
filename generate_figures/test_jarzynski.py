"""Test that the Jarzynski sampler is weighting correctly, by estimating log Z for a case with analytic ground truth."""
import numpy as np
import torch as t
from esh import datasets
import esh
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-paper')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


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

    device = 'cpu'
    d = 3
    eigs = np.clip(np.random.random(d), 0.1, 0.2)
    batch = 10000
    n_steps = 1000
    epsilon = 0.1

    model = datasets.GaussianTest(d, rotate='random', device=device, eig=eigs)
    print('Cov', model.cov)
    print('Eig.', model.eig)
    true_Z_ratio = t.exp(0.5 * model.logdet)
    print('Ground truth, d={}, Z/Z0 = sqrt(det(cov)) = {}, log Z/Z0 = {}'.format(d, true_Z_ratio, np.log(true_Z_ratio)))
    energy = model.energy

    # Initialize
    xs = t.randn(batch, d, device=device)
    vs = t.randn_like(xs)
    v_norm = vs.flatten(start_dim=1).norm(dim=1).view((-1,) + (1,) * (len(vs.shape) - 1))
    vs /= v_norm
    rs = t.zeros(len(xs), device=xs.device, dtype=xs.dtype)  # ESH log |v|
    E0 = 0.5 * t.square(xs).sum(dim=1)
    work = E0 - energy(xs).detach()

    last_grad = None
    print('w mean, w std, rs, energy, H')
    history = np.zeros((n_steps, 7))
    for i in range(n_steps):
        xs, vs, rs, last_grad, _ = esh.esh_leap_step(xs, vs, rs, energy, epsilon, last_grad)  # ESH step with energy scale

        with t.no_grad():
            # d = np.prod(x.shape[1:])
            energies = energy(xs)
            H = energies + d * rs
            # weights = t.exp( rs + log_weight)
            log_weight = E0 - energies + (1-d) * rs
            # log_weight = rs
            weights = t.exp(log_weight)
            # weights = t.exp(E0 - energies)
            #weights = t.exp(E0 - energies + rs)
            history[i] = (weights.mean(), weights.std(), rs.mean(), energies.mean(), H.mean(), log_weight.mean(), log_weight.std())

        print("{:<8.3}  {:<8.3}  {:<8.3}  {:<8.3}  {:<8.3}   {:<8.3}".format(np.log(weights.mean()), weights.std(), rs.mean(), energies.mean(), H.mean(), log_weight.mean()))
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(n_steps), np.log(history[:, 0]), lw=2, label='Jar. estimate')
    ax.axhline(y=np.log(true_Z_ratio), color='r', linestyle='-', label='Ground truth', lw=2)
    # ax.errorbar(np.arange(n_steps), history[:, 5], yerr=history[:, 6], lw=2, label='Lower bound')

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.subplots_adjust(right=0.7)
    ax.set_xlabel('# gradient evaluations')
    ax.set_ylabel('$\log Z/Z_0$')
    fig.savefig('figs/jar.png', transparent=True, bbox_inches="tight")

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(n_steps), history[:, 3], lw=2, label='$\\langle E(x(t)) \\rangle$')
    ax.plot(np.arange(n_steps), d * history[:, 2], lw=2, label='$d~\\langle r(t) \\rangle$')
    ax.plot(np.arange(n_steps), history[:, 5], lw=2, label='$\\langle w(t) \\rangle$')
    ax.plot(np.arange(n_steps), history[:, 4], lw=2, label='$\\langle H(x(t), v(t)) \\rangle$')
    ax.plot(np.arange(n_steps), np.log(history[:, 0]), lw=2, label='$\log \\langle e^{w(t)} \\rangle$')
    ax.axhline(y=np.log(true_Z_ratio), linestyle='-', label='$\log Z/Z_0$', lw=2, alpha=0.3)

    # ax.errorbar(np.arange(n_steps), history[:, 5], yerr=history[:, 6], lw=2, label='Lower bound')

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.subplots_adjust(right=0.7)
    ax.set_xlabel('# gradient evaluations')
    fig.savefig('figs/jar2.png', transparent=True, bbox_inches="tight")