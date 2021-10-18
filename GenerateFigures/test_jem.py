"""Test samplers on the pre-trained energy model from the original JEM paper."""
import os, sys
import math
import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv
import argparse
import time

# viz
import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')  # also try 'seaborn-paper', 'fivethirtyeight'

# local code
import esh

sys.path.append('JEM')
import eval_wrn_ebm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.style.use('seaborn-paper')  # also try 'seaborn-talk', 'fivethirtyeight'
try:
    plt.style.use(os.path.expanduser('~/Dropbox/Public/gv3.mplstyle'))
except:
    pass

t.manual_seed(1)


class ScaleLayer(nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        return input * self.scale


def leap_integrate_wrapper(energy, x0, n_steps, **kwargs):
    epsilon = kwargs['epsilon']
    u0 = kwargs['v']
    xs, vs, rs = esh.leap_integrate_chain(energy, x0, n_steps, epsilon, v=u0)
    return xs, vs, rs


def reservoir_integrate(f, x0, steps, epsilon=0.1, n_binary=0, v=None):
    xs, vs, rs, x_res, u_res, x_res_hist = esh.leap_integrate_chain(f, x0, steps, epsilon, store=True, reservoir=True, v=v)

    for i in range(n_binary):
        epsilon = 0.5 * epsilon
        x, v, r, x_res, u_res, x_res_hist_update = esh.leap_integrate_chain(f, x_res, 2, epsilon, store=True, reservoir=True, v=-u_res)
        x_res_hist = t.cat([x_res_hist, x_res_hist_update], dim=0)

    return x_res_hist, xs, t.arange(len(x_res_hist))


if __name__ == '__main__':

    sampler_list = [
                    #('HMC', samplers.hmc_integrate, {'epsilon': 0.01, 'k': 5, 'mh_reject': True}),
                    #('MALA', samplers.hmc_integrate, {'epsilon': 0.01, 'k': 1, 'mh_reject': True}),
                    #('ULA', samplers.hmc_integrate, {'epsilon': 0.01, 'k': 1, 'mh_reject': False}),
                    #('NH', samplers.nh_integrate, {'epsilon': 0.01}),
                    # ('NH1', samplers.nh_integrate, {'epsilon': 0.1}),
                     # ('NUTS', samplers.nuts, {}),  # can't run on GPU. Giving numerical errors... bc of energy scale?
                    #('ESH-leap1', leap_integrate_wrapper, {'epsilon': 1.}),  # time scaled
                     #('ESH-RK', samplers.esh_rk_integrate, {'rtol': 1e-4, 'atol': 1e-5, 't_scale': True, 'method': 'dopri'}),
                     #('ESH-RK-orig', samplers.esh_rk_integrate, {'rtol': 1e-4, 'atol': 1e-5, 't_scale': False, 'method': 'dopri'}),
                     #('ESH-leap-orig', samplers.leap_unscaled, {'epsilon': 0.01}),  # Not time scaled
        ('Reservoir 0', reservoir_integrate, {'n_binary': 0, 'epsilon': 0.5}),
        # ('Reservoir 3', reservoir_integrate, {'n_binary': 10, 'epsilon': 0.5}),
        ('ESH-leap', leap_integrate_wrapper, {'epsilon': 0.5}),  # time scaled
                   ]

    # Specify shape and placement
    batch = 1
    x_shape = (3, 32, 32)
    d = np.prod(np.array(x_shape))
    n_steps = int(sys.argv[1])
    n_images = int(sys.argv[2])
    prior = sys.argv[3] == 'prior'
    if t.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    xtype = t.float32

    # Load JEM model
    save_dir = os.path.join(os.path.expanduser("~"), 'Dropbox/PycharmProjects/esh_dynamics/JEM')
    model_dir = os.path.join(os.path.expanduser("~"), 'Dropbox/PycharmProjects/esh_dynamics/JEM/CIFAR10_MODEL.pt')
    args = argparse.Namespace(batch_size=64, buffer_size=0, dataset='cifar_test', datasets=[], depth=28, eval='OOD',
                              fresh_samples=False,
                              load_path=model_dir,
                              n_sample_steps=1, n_steps=1, norm=None, ood_dataset='svhn', print_every=100,
                              print_to_log=False, reinit_freq=0.05,
                              save_dir=save_dir,
                              score_fn='px', sgld_lr=1.0, sgld_std=0.01, sigma=0.03, uncond=False, width=10)

    model_cls = eval_wrn_ebm.CCF
    energy = model_cls(args.depth, args.width, args.norm)
    ckpt_dict = t.load(args.load_path)
    energy.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]

    # import IPython; IPython.embed()
    energy = nn.Sequential(energy, ScaleLayer(-20000.))  # JEM uses f to output minus energy, and scales by 20000.
    energy = energy.to(device)
    energy.eval()
    print("Is it in training mode?", energy.training)

    # Turn off parameter gradients, then on again later if training. This totally speeds things up! (from one test)
    if hasattr(energy, 'parameters'):
        save_grad_flags = []  # Turn of grad tracking manually. Can't use context manager because we do autograd inside
        for p in energy.parameters():
            save_grad_flags.append(p.requires_grad)
            p.requires_grad = False

    fig, ax = plt.subplots(1,1)
    for jj in range(1):
        if prior:
            print('prior true')
            x0 = replay_buffer[jj:jj+1].to(device)
            prefix = 'figs/jem_samples_prior/{}'.format(jj)
        else:
            x0 = 2 * t.rand(1, *x_shape, dtype=xtype, device=device) - 1
            prefix = 'figs/jem_samples/{}'.format(jj)
        u0 = t.randn(1, *x_shape, dtype=xtype, device=device)
        u0 = (u0.T / u0.flatten(start_dim=1).norm(dim=1)).T
        for sampler_name, sampler, kwargs in sampler_list:
            print('sampler_name', sampler_name, jj)
            t0 = time.time()
            xs, _, _ = sampler(energy, x0, n_steps, v=u0, **kwargs)
            print('TIME', time.time() - t0, 'steps', n_steps)
            xs = xs[:, 0].view(-1, *x_shape)
            steps_per_grad = int(math.ceil(n_steps / (len(xs) -1)))
            print('steps per', steps_per_grad, n_steps, len(xs), steps_per_grad * len(xs))
            if steps_per_grad > 1:
                xs = xs.repeat_interleave(steps_per_grad, dim=0)

            k = len(xs) // n_images  # Show only n images
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 2))
            ax2.set_aspect('equal')
            ax2.set_xticks([])
            ax2.set_yticks([])
            get_grid = lambda x: tv.utils.make_grid(t.clamp(0.5 * x + 0.5, 0,1), nrow=1).cpu().numpy().transpose((2, 1, 0))
            ax2.imshow(get_grid(xs[::k][:n_images]))
            ax2.set_ylabel(sampler_name, rotation=0, labelpad=40)
            ax2.margins(0, 0)
            ax2.xaxis.set_major_locator(plt.NullLocator())
            ax2.yaxis.set_major_locator(plt.NullLocator())
            fig2.subplots_adjust(left=0.15, right=1, bottom=0, top=1, hspace=0., wspace=0.)
            fig2.savefig('{}jem_{}.png'.format(prefix, sampler_name), pad_inches=0., bbox_inches='tight')
            with t.no_grad():
                energies = energy(xs)
            ax.plot(energies, label=sampler_name)
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        fig.savefig('{}_energy.png'.format(prefix), bbox_inches='tight')