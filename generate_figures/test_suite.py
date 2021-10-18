"""Test a variety of samplers against a variety of datasets. Used to generate MMD plots and ESS table for the paper."""
import os
import sys
import pickle
import torch as t
from torch.utils.tensorboard import SummaryWriter
from esh import utils, samplers, datasets
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-paper')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


if __name__ == '__main__':
    # Seed for reproducibility
    seed = 1
    t.manual_seed(seed)
    if t.cuda.is_available():
        device = t.device('cuda')
        t.cuda.manual_seed_all(seed)
    else:
        device = t.device('cpu')
        print('Warning, no CUDA detected.')

    # Logging
    dataset = 'all'
    log_root = os.path.join(os.path.expanduser("~"), 'tmp/esh/sample_{}/'.format(dataset))
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


    energy_models = [
                      ('2D MOG', datasets.ToyDataset(toy_type='gmm'), 2),
                      ('2D MOG-prior', datasets.ToyDataset(toy_type='gmm'), t.tensor([0., 0.5])),
                      ('50D ICG', datasets.GaussianTest(50, rotate=False, device=device), 50),
                      ('2D SCG', datasets.GaussianTest(2, rotate='strong', device=device), 2),
                      ('2D SCG-bias', datasets.GaussianTest(2, rotate='strong', device=device), 'bias'),
                      ('20D Funnel', datasets.Funnel(), 20)
                     ]


    sampler_list_main = [
                     ('HMC k=5', samplers.hmc_integrate, {'epsilon': 0.01, 'k': 5, 'mh_reject': True}),
                     ('MALA 0.1', samplers.hmc_integrate, {'epsilon': 0.1, 'k': 1, 'mh_reject': True}),
                     ('MALA 0.01', samplers.hmc_integrate, {'epsilon': 0.01, 'k': 1, 'mh_reject': True}),
                     ('ULA 0.1', samplers.hmc_integrate, {'epsilon': 0.1, 'k': 1, 'mh_reject': False}),
                     ('NH', samplers.nh_integrate, {'epsilon': 0.01}),  # Numerical errors above 0.01
                     ('NUTS', samplers.nuts, {}),
                     ('ESH-leap', samplers.leap_integrate, {'epsilon': 0.1}),  # time scaled
                   ]

    sampler_list_ablation = [
                    ('ESH-leap 0.01', samplers.leap_integrate, {'epsilon': 0.01}),  # time scaled
                    ('ESH-leap 0.1', samplers.leap_integrate, {'epsilon': 0.1}),  # time scaled
                    ('ESH-leap-orig 0.01', samplers.leap_unscaled, {'epsilon': 0.01}),  # Not time scaled
                    ('ESH-leap-orig 0.1', samplers.leap_unscaled, {'epsilon': 0.1}),  # Not time scaled
                    ('ESH-RK', samplers.esh_rk_integrate, {'rtol': 1e-4, 'atol': 1e-5, 't_scale': True, 'method': 'dopri'}),
                    ('ESH-RK-orig', samplers.esh_rk_integrate, {'rtol': 1e-4, 'atol': 1e-5, 't_scale': False, 'method': 'dopri'}),
                   ]

    sampler_list_res_test = [
                   ('Reservoir 0', samplers.reservoir_integrate, {'n_binary': 0}),
                   ('Reservoir 3', samplers.reservoir_integrate, {'n_binary': 3}),
                    ('ESH-leap 0.1', samplers.leap_integrate, {'epsilon': 0.1}),  # time scaled
                    ('ULA 0.1', samplers.hmc_integrate, {'epsilon': 0.1, 'k': 1, 'mh_reject': False}),
                   ]


    n_steps = int(sys.argv[1])  # number of gradient steps
    n_repeat = int(sys.argv[2])  # Repeat each experiment
    if sys.argv[3] == 'ablation':
        sampler_list = sampler_list_ablation
        print('Ablation')
    elif sys.argv[3] == 'reservoir':
        sampler_list = sampler_list_res_test
        print("Reservoir test")
    else:
        sampler_list = sampler_list_main
        print('Main sampler list')

    save = []
    for e_name, e_model, x_init in energy_models:
        for sampler_name, sampler, kwargs in sampler_list:
            all_xs, all_vs, all_ts = [], [], []
            for j in range(n_repeat):
                exp_string = '{}_{}_{}_{}'.format(e_name, sampler_name, j, kwargs)
                print(exp_string)
                energy = e_model.energy

                if hasattr(energy, 'parameters'):  # turn off gradient watching for whole script
                    for p in energy.parameters():
                        p.requires_grad = False

                if type(x_init) is int:
                    x0 = t.randn(x_init)  # 2. * t.rand(x_init) - 1.
                elif x_init == 'bias':
                    x0 = t.randn(2) - t.tensor([2., -2.])
                else:
                    x0 = x_init
                xs, vs, ts = sampler(energy, x0, n_steps, **kwargs)
                print('length', len(xs))
                all_xs.append(xs)
                all_vs.append(vs)
                all_ts.append(ts)

                if j == 0:
                    if e_name[:2] == '2D':
                        use_energy = energy
                    else:
                        use_energy = None
                    utils.viz_weighted_trajectory(xs[:, [0, -1]], ts, f=use_energy, weights=True, filename='{}/figs/w_trajectory_{}.png'.format(log_dir, exp_string))
                    utils.viz_weighted_trajectory(xs[:, [0, -1]], ts, f=use_energy, weights=False, filename='{}/figs/trajectory_{}.png'.format(log_dir, exp_string))
                    utils.viz_weighted_trajectory.count += 1

                if sampler_name[:3] == 'ESH' and j == 0:
                    if sampler is samplers.esh_rk_integrate or sampler is samplers.leap_integrate:
                        original_var = sampler.store_diagnostic
                        utils.esh_diagnostics(xs, vs, ts, energy, log_dir, exp_string, original_var)

            save.append((e_name, sampler_name, exp_string, n_repeat, n_steps, all_xs, all_vs, all_ts))
    # Record everything in a pandas dataframe
    # Plot everything
    pickle.dump(save, open(log_dir+'/results.pkl', 'wb'))
    utils.plot_all_ess(save, log_dir)
    utils.plot_all_mmd(save, log_dir)
    # import IPython; IPython.embed()
