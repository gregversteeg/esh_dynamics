##############################
# ## EVALUATE TRAINED EBM ## #
##############################

from collections import OrderedDict
import torch as t
import torchvision.transforms as tr
import torchvision as tv
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import json
import os, sys
import time

from nets import VanillaNet, NonlocalNet
from utils import download_flowers_data, plot_ims
sys.path.append('./inception/')
sys.path.append('..')
import esh


sampler_is_esh = (sys.argv[1][:3] == 'esh')
mcmc_init = sys.argv[2]  # 'uniform' 'buffer'  'gaussian', 'color'
num_mcmc_steps = int(sys.argv[3])
paths = sys.argv[4:]
exp_string = paths[0].split('/')[-3]
# directory for experiment results
EXP_DIR = os.path.expanduser('~/tmp/ebm_anatomy/eval-{}-{}-{}/'.format(sys.argv[1], exp_string, time.strftime("%Y%m%d-%H%M%S")))
print(EXP_DIR)
batch_size = 64
train_epsilon = 0.01
log_freq = max(100, num_mcmc_steps // 500)  # so we get 500 frames for movie
small_log_freq = num_mcmc_steps // 20  # 10 frames for small output

#######################
# ## INITIAL SETUP ## #
#######################

if sampler_is_esh:
    print('Sample with ESH')
    use_mh_langevin = False
else:
    print('Sample with ULA')
    use_mh_langevin = True

# make directory for saving results
if os.path.exists(EXP_DIR):
    # prevents overwriting old experiment folders by accident
    raise RuntimeError('Folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
else:
    os.makedirs(EXP_DIR)
    for folder in ['code']:
        os.mkdir(EXP_DIR + folder)

# save copy of code in the experiment folder
def save_code():
    def save_file(file_name):
        file_in = open('./' + file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)
    for file in ['eval.py', 'nets.py', 'utils.py']:
        save_file(file)
save_code()

# set seed for cpu and CUDA, get device
t.manual_seed(123)
if t.cuda.is_available():
    t.cuda.manual_seed_all(123)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


####################
# ## EVAL SETUP # ##
####################
class SEA(t.nn.Module):
    def __init__(self, f_list):
        """Input list of energy models"""
        super(SEA, self).__init__()
        self.f_list = f_list

    def forward(self, x):
        """Add outputs"""
        for i, f in enumerate(self.f_list):
            if i == 0:
                total = f(x) / len(self.f_list)
            else:
                total += f(x) / len(self.f_list)
        return total


print('Setting up network...')
# set up network and load saved weights
f_list = []
for i, path in enumerate(paths):
    f = VanillaNet(n_c=3, n_f=128)
    sdict = t.load(path, map_location=lambda storage, loc: storage.cpu())
    if t.cuda.device_count() > 1:
        f = t.nn.DataParallel(f)
    else:
        # Strip "module" if DataParallel was used
        sdict = OrderedDict([(k[7:] if k[:7]=='module.' else k, v) for k, v in sdict.items()])
    f.load_state_dict(sdict)
    f.to(device)
    f.eval()
    f_list.append(f)
f = SEA(f_list)
# put net on device
# temperature from training
if train_epsilon > 0:
    temp = (train_epsilon ** 2) / 2
else:
    temp = 1

print('Processing initial MCMC states...')
if mcmc_init == 'uniform':
    q = 2 * t.rand([batch_size, 3, 32, 32]).to(device) - 1
elif mcmc_init == 'gaussian':
    q = t.randn([batch_size, 3, 32, 32]).to(device)
elif mcmc_init == 'color':
    mu = t.tensor([-0.0172, -0.0357, -0.1069])  # average image color
    M = t.tensor([[0.2181, 0.1126, 0.0751],
                  [0.1126, 0.1878, 0.1238],
                  [0.0751, 0.1238, 0.2703]])  # sqrt of covariance of average image color
    q = mu.reshape((1, 3)) + t.matmul(t.randn(batch_size,3), M)
    q = 0.5 * q.reshape((batch_size, 3, 1, 1)) + 0.5 * t.randn((batch_size, 3, 32, 32))
    q = q.to(device)
else:
    # use buffer
    q = t.load(paths[-1].replace('net', 'buffer'))
    q = q.to(device)
q = t.clamp(q, -1, 1)

# get a random sample of initial states from image bank
x_s_t_0 = q[t.randperm(q.shape[0])[0:batch_size]]


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# langevin equation without MH adjustment
def langevin_grad():
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)

    # sampling records
    grads = t.zeros(num_mcmc_steps + 1, batch_size)
    ens = t.zeros(num_mcmc_steps + 1, batch_size)
    small_output = []

    # iterative langevin updates of MCMC samples
    for ell in range(num_mcmc_steps + 1):
        en = f(x_s_t) / temp
        ens[ell] = en.detach().cpu()
        if ell % log_freq == 0 or ell == num_mcmc_steps:
            print('Step {} of {}.   Ave. En={:>14.9f} '.
                  format(ell, num_mcmc_steps, ens[ell].mean()))
            plot_ims(EXP_DIR + 'sample_states_{:05d}.png'.format(ell), x_s_t.detach())
        if ell % small_log_freq == 0 or ell == num_mcmc_steps:
            small_output.append(x_s_t.detach().clone()[:8])

        grad = t.autograd.grad(en.sum(), [x_s_t])[0]
        if train_epsilon > 0:
            v = t.randn_like(x_s_t)
            x_s_t.data += - ((train_epsilon**2)/2) * grad + train_epsilon * v
            # x_s_t.data, v = esh.billiards(x_s_t.data, v)  # Implement boundary constraint
            grads[ell] = ((train_epsilon**2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        else:
            x_s_t.data += - grad
            grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()

    return x_s_t.detach(), ens, grads, small_output

# langevin equation with MH adjustment
def langevin_mh():
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)

    # sampling records
    ens = t.zeros(num_mcmc_steps, batch_size)
    grads = t.zeros(num_mcmc_steps, batch_size)
    accepts = t.zeros(num_mcmc_steps)

    # iterative langevin updates of MCMC samples
    for ell in range(num_mcmc_steps):
        # get energy and gradient of current states
        en = f(x_s_t) / temp
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t])[0]
        grads[ell] = ((train_epsilon ** 2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()

        # get initial gaussian momenta
        p = t.randn_like(x_s_t)

        # get proposal states
        x_prop = x_s_t - ((train_epsilon ** 2)/2) * grad + train_epsilon * p
        # update momentum
        en_prop = f(x_prop) / temp
        grad_prop = t.autograd.grad(en_prop.sum(), [x_prop])[0]
        p_prop = p - (train_epsilon / 2) * (grad + grad_prop)

        # joint energy of states and auxiliary momentum variables
        joint_en_orig = en + 0.5 * t.sum((p ** 2).view(x_s_t.shape[0], -1), 1)
        joint_en_prop = en_prop + 0.5 * t.sum((p_prop ** 2).view(x_s_t.shape[0], -1), 1)

        # accept or reject states_prop using MH acceptance ratio
        accepted_proposals = t.rand_like(en) < t.exp(joint_en_orig - joint_en_prop)

        # update only states with accepted proposals
        x_s_t.data[accepted_proposals] = x_prop.data[accepted_proposals]
        accepts[ell] = float(accepted_proposals.sum().cpu()) / float(batch_size)

        if ell == 0 or (ell + 1) % log_freq == 0 or (ell + 1) == num_mcmc_steps:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}   Accept Rate={:>14.9f}'.
                  format(ell+1, num_mcmc_steps, ens[ell].mean(), grads[ell].mean(), accepts[ell]))
        # if ell % 10000 == 0:
            plot_ims(EXP_DIR + 'sample_states_{:05d}.png'.format(ell+1), x_s_t.detach())

    return x_s_t.detach(), ens, grads, accepts

def esh_sample():
    ens = t.zeros(num_mcmc_steps, batch_size)
    epsilon = 0.5
    small_output = []
    r = t.zeros(len(x_s_t_0), dtype=x_s_t_0.dtype, device=x_s_t_0.device)
    u = t.randn_like(x_s_t_0)
    u /= u.flatten(start_dim=1).norm(dim=1).view((len(x_s_t_0),) + (1,) * (len(x_s_t_0.shape) - 1))
    x_s_t = x_s_t_0.clone()
    L = 100
    for ell in range(num_mcmc_steps // L + 1):
        en = f(x_s_t) / temp
        ens[ell] = en.detach().cpu()
        if (ell * L) % log_freq == 0 or ell == (num_mcmc_steps // L):
            print('Step {} of {}.   Ave. En={:>14.9f}  '.
                  format(ell * L, num_mcmc_steps, ens[ell].mean()))
            plot_ims(EXP_DIR + 'sample_states_{:05d}.png'.format(ell*L), x_s_t.detach())
        if (ell * L) % small_log_freq == 0 or ell == (num_mcmc_steps // L):
            small_output.append(x_s_t.detach().clone()[:8])

        # reservoir sample at each batch, as in training
        x_b, v_b, r_b, x_s_t, u_res = esh.leap_integrate_chain(f, x_s_t, L, epsilon, store=False, reservoir=True, energy_scale=1. / temp)  # , constraint=True)
        # Or ESH Dynamics directly
        # x_s_t, u, r_b, x_s_t, u_res = esh.leap_integrate_chain(f, x_s_t, L, epsilon, store=False, reservoir=True, energy_scale=1. / temp, v=u)
    return x_s_t.detach(), ens, small_output


###################################
# ## SAMPLE FROM LEARNED MODEL ## #
###################################

print('Sampling for {} steps.'.format(num_mcmc_steps))
if use_mh_langevin:
    x_s_t, en_record, grad_record, small_output = langevin_grad()  # , accept_record = langevin_mh()
    # plt.plot(accept_record.numpy())
    # plt.savefig(EXP_DIR + 'accept.png')
    # plt.close()
    plt.plot(grad_record.numpy())
    plt.title('Gradient magnitude over sampling path')
    plt.xlabel('Langevin step')
    plt.ylabel('Gradient magnitude')
    plt.savefig(EXP_DIR + 'grad.png')
    plt.close()
else:
    x_s_t, en_record, small_output = esh_sample()

# # visualize initial and synthesized images
# plot_ims(EXP_DIR + 'initial_states.png', x_s_t_0[:100])
# plot_ims(EXP_DIR + 'final_states.png', x_s_t[:100])

# plot diagnostics
# plt.plot(en_record.numpy())
# plt.title('Energy over sampling path')
# plt.xlabel('Langevin step')
# plt.ylabel('energy')
# plt.savefig(EXP_DIR + 'en.png')
# plt.close()

small_output = t.stack(small_output, dim=1).reshape((-1, 3, 32,32))
tv.utils.save_image(t.clamp(small_output, -1., 1.), EXP_DIR + 'small.png', normalize=True, nrow=len(small_output) // 8)

# def plot_ims(p, x): tv.utils.save_image(t.clamp(x, -1., 1.), p, normalize=True, nrow=int(x.shape[0] ** 0.5))
# import IPython; IPython.embed()
# IS, FID = get_inception_score_and_fid(t.clamp(0.5 * (1. + x_s_t), 0, 1), './cifar10.test.npz', verbose=True)
# print(IS, FID)