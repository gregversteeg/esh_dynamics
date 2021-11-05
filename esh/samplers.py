"""
Energy Sampling Hamiltonian Dynamics Sampler Tests and Baselines

For baselines, we implement a number of Markov chain samplers that use gradients, i.e.
HMC, Langevin, Nose-Hoover

A word of explanation is in order in case anyone ever scrutinizes this library. There appears to be a
very dumb design choice: each method can only run a single sampler chain at a time, not several in parallel.
The reason for this is that I was originally focused on using adaptive Runge-Kutta for ESH sampling. There's no
easy way to adapt Chen et al's code to have parallel chains, each with their own step size.
Then, because I was doing one chain at a time, I did all the other baselines the same way.
Later, I wrote my own parallel version of Runge-Kutta, where each chain could have it's own step size and time variable
Then, I realized that Runge-Kutta integration is garbage compared to a specialized leapfrog integrator for the
time-scaled ESH ODE that I discovered.
But I never went back and fixed up this code to be more parallel. I only use it for toy benchmarks anyway. So
I don't actually recommend using this code for anything.
For ESH Leapfrog integrators, directly use esh.py functions.
"""
import numpy as np
import torch as t
import torch.nn as nn
from torchdiffeq import odeint
from torchdiffeq._impl.dopri5 import Dopri5Solver
from torchdiffeq._impl.bosh3 import Bosh3Solver
from torchdiffeq._impl.misc import _PerturbFunc, _rms_norm
from torchdiffeq._impl.interp import _interp_evaluate
import littlemcmc
import esh
np.random.seed(1)


def v2_d(vec):
    return t.square(vec).mean()  # v^2 / d shorthand


def l2_norm_flat(vec):
    """Shorthand for squared L2 norm per batch dimension, flattening all dimensions."""
    return t.einsum('j,j', vec.flatten(start_dim=0), vec.flatten(start_dim=0))


def nuts(f, x0, steps):
    """NUTS wrapper https://github.com/eigenfoo/littlemcmc
    We can't control the number of gradient steps directly.
    We'll over-shoot and then cut off at the appropriate point."""
    def g(x):
        g.counter += 1
        logp = f(t.tensor(x, dtype=x0.dtype))
        x = t.autograd.Variable(t.tensor(x, dtype=x0.dtype).clone().detach(), requires_grad=True)
        grad = t.autograd.grad(f(x).sum(), [x])[0].detach()
        return -logp.numpy(), -grad.numpy()
    g.counter = 0

    trace, stats = littlemcmc.sample(logp_dlogp_func=g, model_ndim=len(x0), progressbar=None, tune=0, chains=1,
                                     start=x0.numpy(), draws=steps)
    # "tune" introduces gradient overhead that makes it completely un-competitive
    nut_xs = trace[0]
    ts = np.cumsum(stats['tree_size'].flatten() + 1)   # t corresponds to number of gradients used
    nut_xs = np.insert(nut_xs, 0, x0.numpy(), axis=0)
    ts = np.insert(ts, 0, 0)

    shape = x0.shape
    xs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    for i in range(steps+1):
        nuts_index = np.where(ts <= i)[0][-1]  # nuts step that uses less than or equal to i gradients
        xs[i] = t.tensor(nut_xs[nuts_index])

    print('Number of gradients for NUTS {}. Looking for {}. Used {} steps'.format(g.counter, steps, nuts_index))
    return xs, t.zeros_like(xs), t.arange(len(xs))  # Usually I return velocity - we don't have it so I return 0s


# HMC functions
def leapfrog_step(x_0, v_0, f, epsilon, k, mh_reject=False):
    """A single leapfrog step with energy, f, and optional energy scaling and bounding box (at +/1)"""
    x_k = t.autograd.Variable(x_0.clone(), requires_grad=True)
    v_k = v_0.clone()
    for _ in range(k):  # Inefficient version - should combine half steps
        v_k -= 0.5 * epsilon * t.autograd.grad(f(x_k).sum(), [x_k])[0]  # half step in v
        x_k.data += epsilon * v_k.detach()  # Step in x
        grad = t.autograd.grad(f(x_k).sum(), [x_k])[0]
        v_k -= 0.5 * epsilon * grad  # half step in v
    if mh_reject:
        with t.no_grad():
            delta_v = 0.5 * (l2_norm_flat(v_0) - l2_norm_flat(v_k))
            delta_joint_E = f(x_0) - f(x_k) + delta_v
            reject = (np.random.random() > t.exp(delta_joint_E).item())
            if reject:
                return x_0, v_0
    return x_k.detach(), v_k.detach()


def hmc_integrate(f, x0, steps, epsilon=0.01, k=1, mh_reject=False):
    shape = x0.shape
    grad_steps = steps // k
    xs = t.zeros((grad_steps + 1,) + shape, dtype=x0.dtype)
    vs = t.zeros((grad_steps + 1,) + shape, dtype=x0.dtype)
    xs[0] = x0
    vs[0] = t.randn_like(x0)
    x = x0
    for l in range(1, grad_steps + 1):
        x, v = leapfrog_step(x, t.randn_like(x), f, epsilon, k, mh_reject=mh_reject)
        xs[l], vs[l] = x, v
    return xs, vs, t.arange(len(xs))


def newton_dynamics(f, x0, steps, epsilon=0.01):
    """Just do Newtonian Hamiltonian dynamics, no momentum randomization and no MH reject."""
    shape = x0.shape
    xs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    vs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    v = t.randn_like(x0)
    x = x0
    xs[0], vs[0] = x, v
    for l in range(1, steps + 1):
        x, v = leapfrog_step(x, v, f, epsilon, 1, mh_reject=False)
        xs[l], vs[l] = x, v
    return xs, vs


# ESH Leap functions
def klog_step(x, v, f, epsilon):
    """Take a single Hamilton step for K = d/2 log v^2/d."""
    x = t.autograd.Variable(x.clone(), requires_grad=True)
    v = v.clone()
    grad = t.autograd.grad(f(x).sum(), [x])[0]
    v -= 0.5 * epsilon * grad  # half step in v
    x.data += epsilon * v  # Full step in x
    grad = t.autograd.grad(f(x).sum(), [x])[0]
    klog_step.last_grad = grad
    v -= 0.5 * epsilon * grad
    return x.detach(), v, epsilon


def leap_unscaled(f, x0, steps, epsilon=0.1):
    # make vector to store info: time, x, v
    ts = t.zeros(steps+1, dtype=t.float32)
    shape = x0.shape
    xs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    vs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    x, v = x0, t.randn_like(x0)
    xs[0], vs[0] = x, v
    for i in range(steps):
        x, v, dt = klog_step(x, v, f, epsilon)
        xs[i+1], vs[i+1] = x, v
        ts[i+1] = ts[i] + dt
    return xs, vs, ts


# Nose Hoover methods
def hoover_step(x, v, alpha, eta, f, eps, m=1):
    """A single step for Nose-Hoover dynamics, using the Kleinerman 08 variation"""
    dof = len(v.flatten(start_dim=0))

    v = v.clone()
    alpha += eps / 4. * (l2_norm_flat(v) - dof)
    eta += eps / 2. * m * alpha
    v *= t.exp(-m * eps / 2. * alpha)
    alpha += eps / 4. * (l2_norm_flat(v) - dof)
    x = t.autograd.Variable(x.clone(), requires_grad=True)
    v -= 0.5 * eps * t.autograd.grad(f(x).sum(), [x])[0]  # TODO: store grad from last iteration, if possible
    x.data += eps * v  # Full step in x
    grad = t.autograd.grad(f(x).sum(), [x])[0]
    v -= 0.5 * eps * grad
    x = x.detach()
    alpha += eps / 4. * (l2_norm_flat(v) - dof)
    v *= t.exp(-m * eps / 2. * alpha)
    eta += eps / 2. * m * alpha
    alpha += eps / 4. * (l2_norm_flat(v) - dof)
    return x, v, alpha, eta


def nh_integrate(energy, x0, steps, epsilon=0.1):
    ts = t.zeros(steps+1, dtype=t.float32)
    shape = x0.shape
    xs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    vs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    x, v = x0, t.randn_like(x0)
    xs[0], vs[0] = x, v
    alpha, eta = 0., 0.
    for i in range(steps):
        x, v, alpha, eta = hoover_step(x, v, alpha, eta, energy, epsilon)
        xs[i+1], vs[i+1] = x, v
        ts[i+1] = ts[i] + 1.
    return xs, vs, ts


# ESH Adaptive solver
def esh_rk_integrate(energy, x0, n_steps, rtol=1e-3, atol=1e-4, t_scale=True,
                     device='cpu', method='dopri', interp=False):
    """Call the internals of Chen's ODE solver to get exactly some number of steps,
    but still interpolate onto a constant time grid, which is convenient for analysis and visualization.
    t_scale: whether to solve the ESH dynamics directly, or the scaled time dynamics. Need to re-scale back at the end
    so that we can do ergodic sampling.
    """
    if hasattr(energy, 'parameters'):
        save_grad_flags = []  # Turn of grad tracking manually. Can't use context manager because we do autograd inside
        for p in energy.parameters():
            save_grad_flags.append(p.requires_grad)
            p.requires_grad = False

    if t_scale:
        def g(_, y):
            """Scaled ESH ODE grad call. The function passed to the ODE solver has size (2,) + x_shape."""
            g.counter += 1  # Keep track of calls, gradient is most expensive part of computation
            x = t.autograd.Variable(y[0].clone(), requires_grad=True)
            grad = t.autograd.grad(energy(x).sum(), [x])[0].detach()
            v_mag = t.sqrt(t.square(y[1]).sum())
            d = y[1].flatten().shape[0]
            return t.stack([y[1] / v_mag, -grad * v_mag / d])
    else:
        def g(_, y):
            """Unscaled ESH ODE grad call. The function passed to the ODE solver has size (2,) + x_shape."""
            g.counter += 1  # Keep track of calls, gradient is most expensive part of computation
            x = t.autograd.Variable(y[0].clone(), requires_grad=True)
            grad = t.autograd.grad(energy(x).sum(), [x])[0].detach()
            return t.stack([y[1] / v2_d(y[1]), -grad])
    g.counter = 0
    gp = _PerturbFunc(g)  # Required wrapper for torchdiffeq library

    y0 = t.stack([x0, t.randn_like(x0)])  # TODO: scale v?
    if method == 'dopri':
        solver = Dopri5Solver(func=gp, y0=y0, rtol=rtol, atol=atol, norm=_rms_norm)
        grads_per_step = 6
    elif method == 'bosh':
        solver = Bosh3Solver(func=gp, y0=y0, rtol=rtol, atol=atol, norm=_rms_norm)
        grads_per_step = 3
    elif method == 'implicit':
        solver = AdamsBashforthMoulton(func=gp, y0=y0)
    solver._before_integrate([t.tensor(0., dtype=t.float64, device=device)])

    steps = n_steps // grads_per_step
    # make vectors to store info: time, x, v
    ts = t.zeros(steps+1, dtype=t.float64)
    shape = y0[0].shape
    xs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    vs = t.zeros((steps + 1,) + shape, dtype=x0.dtype)
    interp_coeffs = t.zeros((steps, 5, 2,) + shape, dtype=solver.rk_state.interp_coeff[0].dtype)  # Store interpolation coefficients
    xs[0] = y0[0].cpu()
    vs[0] = y0[1].cpu()
    for i in range(steps):
        solver.rk_state = solver._adaptive_step(solver.rk_state)
        xs[i+1] = solver.rk_state.y1[0].cpu()
        vs[i+1] = solver.rk_state.y1[1].cpu()
        ts[i+1] = solver.rk_state.t1.cpu()
        interp_coeffs[i] = t.stack(solver.rk_state.interp_coeff)

    esh_rk_integrate.store_diagnostic = (xs, vs, ts)  # Stored so I can access for diagnostics
    if hasattr(energy, 'parameters'):
        for p in energy.parameters():
            p.requires_grad = save_grad_flags.pop(0)  # Restore original requires_grad flags

    if interp:
        # Interpolate to a regular time grid
        resolution = 20  # how finely to interpolate the temporal mesh
        new_ts = t.linspace(0., ts[-1] - 1e-7, resolution * steps, dtype=t.float64)
        new_xs = t.zeros((resolution * steps,) + shape, dtype=x0.dtype)
        new_vs = t.zeros((resolution * steps,) + shape, dtype=x0.dtype)
        new_xs[0] = xs[0]
        new_vs[0] = vs[0]
        i = 0  # location of next time point
        # print(ts)
        for j, next_t in enumerate(new_ts[1:]):
            # Find i so that next_t is between ts[i], ts[i+1]
            # print(next_t, ts[i], i)
            while next_t.item() > ts[i].item() and i < len(ts) - 1:
                i += 1
            if next_t.item() <= ts[i].item():
                z = _interp_evaluate(interp_coeffs[i-1], ts[i-1], ts[i], next_t)
                new_xs[j+1], new_vs[j+1] = z[0], z[1]
            else:
                print('Interpolating mismatch at i={}, j={}, next_t={}, t[-1]={}'.format(i,j, next_t, ts[-1]))

        if t_scale:  # Now scale to original ts...
            dt = new_ts[1]
            v_mag_d = t.sqrt(t.square(new_vs).sum(axis=1)) / len(vs[0].flatten())
            new_ts = dt * t.cumsum(v_mag_d, 0)
            new_ts = new_ts - new_ts[0]
            # Linear interpolate back to regular spaced grid, Needed for autocorrelation / ESS
            # Do this only within ESS calculation
            # new_xs, _ = lin_interp(new_xs, new_ts)
            # new_vs, new_ts = lin_interp(new_vs, new_ts)

        new_ts *= float(n_steps) / new_ts[-1]  # Time-scale is meaningless, so we normalize by number of gradients
        return new_xs, new_vs, new_ts
    else:
        if t_scale:  # Now scale to original ts...
            dt = ts[1]
            v_mag_d = t.sqrt(t.square(vs).sum(axis=1)) / len(vs[0].flatten())
            new_ts = dt * t.cumsum(v_mag_d, 0)
            new_ts = new_ts - new_ts[0]
            return xs, vs, new_ts
        else:
            return xs, vs, ts


def leap_integrate(f, x0, steps, epsilon=0.01, energy_scale=1.):
    """Integrate a single chain using time-scaled ESH dynamics solved via leapfrog,
    starting at x0, using "steps" grad evals."""
    xs, vs, rs = esh.leap_integrate_chain(f, x0[None, ...], steps, epsilon, energy_scale=energy_scale)
    xs, vs = xs[:,0], vs[:,0]
    rs = rs - rs.max() + 10  # only defined up to constant anyway, we want to keep it from overflowing exp
    # ts = t.cumsum(t.exp(rs[:, 0]), dim=0)  # Riemann sum
    # ts = ts - ts[0]
    ts = t.cat([t.zeros(1), t.cumsum((t.exp(rs[:-1, 0]) + t.exp(rs[1:, 0])) * 0.5, dim=0)], dim=0)  # Trapezoidal rule
    ts = ts * float(steps) / ts[-1]  # Time linear scale is meaningless, so we normalize by number of gradients
    vs = vs * t.exp(rs).view((-1, 1))  # Change from u (unit vector) to v (with magnitude)
    leap_integrate.store_diagnostic = (xs, vs, ts)  # Store to look at with diagnostics
    return xs, vs, ts


def reservoir_integrate(f, x0, steps, epsilon=0.1, n_binary=0):
    xs, vs, rs, x_res, u_res, x_res_hist = esh.leap_integrate_chain(f, x0[None, ...], steps, epsilon, store=True, reservoir=True)

    for i in range(n_binary):
        epsilon = 0.5 * epsilon
        x, v, r, x_res, u_res, x_res_hist_update = esh.leap_integrate_chain(f, x_res, 2, epsilon, store=True, reservoir=True, v=-u_res)
        x_res_hist = t.cat([x_res_hist, x_res_hist_update], dim=0)

    return x_res_hist[:,0], xs[:,0], t.arange(len(x_res_hist))


def random_rotate_vector(v, alpha):
    """Rotate v around some random orthogonal vector c, by angle alpha."""
    c = t.randn_like(v)
    v2 = l2_norm_flat(v)
    c_dot_v = (c * v).sum()
    c -= c_dot_v / v2 * v
    c *= t.sqrt(v2) / t.sqrt(l2_norm_flat(c))
    v_prime = np.cos(alpha) * v + np.sin(alpha) * c
    return v_prime
