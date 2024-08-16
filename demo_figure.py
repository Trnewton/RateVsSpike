'''Script to generate data and plots for figure 3.

Run
    >>> python demo_figure.py
to generate data for and plot figure 3 (WARNING: this will take a long time) or
    >>> python demo_figure.py --help
for help.
'''


from collections import namedtuple
import pickle
from typing import Tuple

import numpy as np
from scipy.integrate import odeint
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

from pysitronics import networks as nn
from pysitronics import optimization as opt
import supervisors as sups


Data = namedtuple('Data', ['LIF_warm', 'Rate_warm','LIF_train', 'Rate_train', 'LIF_test', 'Rate_test'])

NEURON_PARAMS = {
    't_m' : 0.01, # Membrane time constant (s)
    't_ref' : 0.002, # Refractory time constant (s)
    'v_reset' : -65, # Reset voltage (mV)
    'v_peak' : -40, # Peak voltage (mV)
    'i_bias': -40, # Bias current (mV)
    't_r' : 0.002, # Rise  time (s)
    't_d' : 0.02, # Decay time (s)
}
DT = 5e-5
TITLE_FONT_SIZE = 34
SCALE_FONT_SIZE = 28
SUP_COLOUR = '0.6'


## Utility functions ##

def train_networks(
        sup_train: np.ndarray,
        sup_test: np.ndarray,
        N: int,
        g: float,
        p: float,
        q: float,
        dim: int,
        row_balance: bool,
        neuron_params: dict,
        alpha: float,
        learn_step: int,
        warmup_steps: int,
        learn_reps: int,
        test_reps: int,
        train_saves: dict={},
        test_saves: dict={},
        I_in_train: np.ndarray=None,
        I_in_test: np.ndarray=None
    ) -> Data:
    '''Trains network with given specifications to produce given supervisor.'''

    # Create network connectivity matrices
    omega = nn.omega_gen(N, g, p)
    if row_balance:
        nn.row_balance(omega)
    eta = nn.eta_gen(N, q, dim=dim)
    # Input weight matrix
    psi = nn.eta_gen(N, q, dim=dim)
    # Make Networks
    LIF_net = nn.LIF(
        N=N,
        dim=dim,
        omega=omega,
        eta=eta,
        psi=psi,
        **neuron_params
    )
    Rate_net = nn.LIF_Rate(
        N=N,
        dim=dim,
        omega=omega,
        eta=eta,
        psi=psi,
        **neuron_params
    )

    # Make optimizer and train networks
    optimizer = opt.FORCE(
        alpha,
        sup_train,
        DT,
        warmup_steps,
        sup_train.shape[0] * learn_reps,
        learn_step,
        I_in_train
    )

    LIF_warmup, LIF_train = optimizer.teach_network(LIF_net, train_saves)
    Rate_warmup, Rate_train = optimizer.teach_network(Rate_net, train_saves)

    # Test
    x_hat_test_LIF, LIF_test = LIF_net.simulate(DT, sup_test.shape[0]*test_reps, I_in_test, test_saves)
    x_hat_test_Rate, Rate_test = Rate_net.simulate(DT, sup_test.shape[0]*test_reps, I_in_test, test_saves)

    LIF_test['x_hat'] = x_hat_test_LIF
    Rate_test['x_hat'] = x_hat_test_Rate

    return Data(LIF_warmup, Rate_warmup, LIF_train, Rate_train, LIF_test, Rate_test)

def save_data(data: Data, file: str):
    '''Saves data to given file.'''
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_date(file: str) -> Data:
    '''Loads data from given file.'''
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


## Supervisor generation functions ##

def lorenz_deriv(state, t: float, rho: float, sigma: float, beta: float, kappa: float):
    '''Computes the derivatives for the Lorenz system.'''
    x, y, z = state  # Unpack the state vector
    return kappa*sigma * (y - x), kappa*(x * (rho - z) - y), kappa*(x * y - beta * z)  # Derivatives

def gen_lorenz(t_end: float, dt: float, rho: float=28.0, sigma: float=5.0, beta: float=7.0/3.0, kappa: float=1):
    '''Generates the Lorenz supervisor.'''

    lorenz_args = (rho, sigma, beta, kappa)
    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0, t_end, dt)
    states = odeint(lorenz_deriv, state0, t, args=lorenz_args)

    return states

def sin_sup(T: np.ndarray, fq: float=5, A: float=1):
    '''Generates sin supervisor.'''
    return A * np.sin(2 * np.pi * fq * T)

def oscillator_sup(T: np.ndarray, fq=np.array([1, 2]), A=np.array([1, 1])):
    '''Generates oscillator supervisor.'''
    return A @ np.sin(2 * np.pi * np.outer(fq, T))

## Data generation functions ##
def sin_data_gen() -> Data:
    '''Generates data for sin panel.'''

    # Signal
    fq = 5
    A = 1
    # Steps in supervisor
    steps_in_period = int(1 / (DT * fq))
    T = np.arange(0, steps_in_period * DT, DT)
    sup = sin_sup(T, fq, A)

    # Network parameters
    N = 2000
    dim = 1
    p = 0.4
    q = 25
    g = 0.19
    row_balance = True

    # Optimization parameters
    warmup_steps = 3 * sup.shape[0]
    training_reps = 5
    learn_step = 50
    alpha = 0.1 * DT
    train_saves = {
        'x_hat': None,
        'phi' : N,
        'R' : 10
    }

    # Testing parameters
    test_reps = 3
    test_saves = {
        'phi' : N,
        'R' : 10
    }

    # Run simulations
    return train_networks(
        sup,
        sup,
        N,
        g,
        p,
        q,
        dim,
        row_balance,
        NEURON_PARAMS,
        alpha,
        learn_step,
        warmup_steps,
        training_reps,
        test_reps,
        train_saves,
        test_saves
    )

def pitchfork_data_gen() -> Data:
    '''Generates data for pitckfork panel.'''

    dim = 1
    # Pitchfork time constant
    gamma = 0.01

    # Perterbation parametesr
    min_pulse_lenght = int(0.005/DT)
    max_pulse_lenght = int(0.01/DT)

    min_start_time = int(0.1/DT)
    max_start_time = int(0.5/DT)

    mean_pulse_height = 2
    num_pulse = 400

    sup_train, pert_train = sups.pitchfork(DT, gamma, min_pulse_lenght,
        max_pulse_lenght, min_start_time, max_start_time, mean_pulse_height,
        num_pulse, seed=2)
    sup_test, pert_test = sups.pitchfork(DT, gamma, min_pulse_lenght,
        max_pulse_lenght, min_start_time, max_start_time, mean_pulse_height,
        num_pulse, seed=4)

    # Network parameters
    N = 2000
    dim = 1
    p = 0.4
    q = 28
    g = 0.16
    row_balance = True

    # Optimization parameters
    warmup_steps = 2000
    training_reps = 1
    learn_step = 100
    alpha = 0.1 * DT
    train_saves = {}

    # Testing parameters
    test_reps = 1
    test_saves = {'R' : 10}

    # Run simulations
    return train_networks(
        sup_train, sup_test, N, g, p, q, dim, row_balance, NEURON_PARAMS, alpha,
        learn_step, warmup_steps, training_reps, test_reps, train_saves,
        test_saves, pert_train, pert_test
    )

def oscillator_data_gen() -> Data:
    '''Genderates data for oscillator panel.'''

    # Signal
    fq = np.array([1, 2])
    A = np.array([1, 1])
    # Steps in supervisor
    steps_in_period = int(1 / (DT * np.min(fq)))
    T = np.arange(0, steps_in_period * DT, DT)
    sup = oscillator_sup(T, fq, A)

    # Network parameters
    N = 2000
    dim = 1 if sup.ndim == 1 else sup.shape[1]
    p = 0.3
    q = 20
    g = 0.1
    row_balance = True

    # Optimization parameters
    warmup_steps = 2000
    training_reps = 5
    learn_step = 100
    alpha = 0.1 * DT
    train_saves = {}

    # Testing parameters
    test_reps = 2
    test_saves = {'R' : 10}

    # Run simulations
    return train_networks(
        sup, sup, N, g, p, q, dim, row_balance, NEURON_PARAMS, alpha, learn_step,
        warmup_steps, training_reps, test_reps, train_saves, test_saves
    )

def ode_data_gen() -> Data:
    '''Generates data for Ode to Joy panel.'''

    # Signal
    sup, _ = sups.get_sup('ode_short_HDTS', DT)

    # Network parameters
    N = 2000
    dim = sup.shape[1]
    p = 0.4
    q = 25
    g = 0.16
    row_balance = True

    # Optimization parameters
    warmup_steps = 2000
    training_reps = 20
    learn_step = 100
    alpha = 0.1 * DT
    train_saves = {}

    # Testing parameters
    test_reps = 2
    test_saves = {'R' : 10}

    # Run simulations
    return train_networks(
        sup, sup, N, g, p, q, dim, row_balance, NEURON_PARAMS, alpha, learn_step,
        warmup_steps, training_reps, test_reps, train_saves, test_saves
    )

def lorenz_data_gen() -> Data:
    '''Generates data for Lorenz panel.'''

    # Signal
    sup = gen_lorenz(200, DT)

    # Network parameters
    N = 2000
    dim = sup.shape[1]
    p = 0.3
    q = 20
    g = 0.1
    row_balance = True
    neuron_params = NEURON_PARAMS.copy()
    neuron_params['t_d'] = 0.1

    # Optimization parameters
    warmup_steps = 2000
    training_reps = 1
    learn_step = 100
    alpha = 0.1 * DT
    train_saves = {}

    # Testing parameters
    sup_test = sup[:int(sup.shape[0]/2),:]
    test_reps = 1
    test_saves = {'R' : 10}

    # Run simulations
    return train_networks(
        sup, sup_test, N, g, p, q, dim, row_balance, neuron_params, alpha, learn_step,
        warmup_steps, training_reps, test_reps, train_saves, test_saves
    )


## Plotting functions ##

def sin_plot(data: Data, save_dir: str=None, neuron_idx: list=None):
    '''Plots sin panel.'''

    #### Process data ####
    R_LIF = np.vstack([data.LIF_warm['R'], data.LIF_train['R'], data.LIF_test['R']])
    R_Rate = np.vstack([data.Rate_warm['R'], data.Rate_train['R'], data.Rate_test['R']])
    if neuron_idx:
        num_neuron = len(neuron_idx)
    else:
        if not R_LIF.shape == R_Rate.shape:
            raise ValueError('LIF-spike and LIF-rate neuron time series are not the same shape.')
        num_neuron = R_Rate.shape[1]
        neuron_idx = np.arange(num_neuron)

    x_hat_LIF = np.concatenate([
        data.LIF_warm['x_hat'],
        data.LIF_train['x_hat'],
        data.LIF_test['x_hat']
    ])
    x_hat_Rate = np.concatenate([
        data.Rate_warm['x_hat'],
        data.Rate_train['x_hat'],
        data.Rate_test['x_hat']
    ])

    warmup_steps = data.Rate_warm['x_hat'].size
    train_steps = data.Rate_train['x_hat'].size
    eval_steps = data.Rate_test['x_hat'].size
    full_steps = warmup_steps + train_steps + eval_steps

    T = np.arange(0, DT * full_steps, DT)
    sup = sin_sup(T)

    learn_step = 50

    phi_norm_LIF = np.linalg.norm(data.LIF_train['phi'], axis=1)[::learn_step]
    phi_norm_LIF_Rate = np.linalg.norm(data.Rate_train['phi'], axis=1)[::learn_step]

    d_phi_norm_LIF = phi_norm_LIF[1:] - phi_norm_LIF[:-1]
    d_phi_norm_LIF_Rate = phi_norm_LIF_Rate[1:] - phi_norm_LIF_Rate[:-1]

    d_phi_norm_LIF = np.concatenate(
        [np.zeros(warmup_steps//learn_step),
        d_phi_norm_LIF,
        np.zeros(eval_steps//learn_step)]
    )
    d_phi_norm_LIF_Rate = np.concatenate([
        np.zeros(warmup_steps//learn_step),
        d_phi_norm_LIF_Rate,
        np.zeros(eval_steps//learn_step)]
    )
    T_phi = np.arange(0, DT * full_steps, DT * learn_step)[:-1]


    #### Plotting ####
    gs_kwarg = {'height_ratios' : [1 for n in range(num_neuron)]+[1,1,0.2]}
    _, axes = plt.subplots(nrows=3+num_neuron, ncols=1, figsize=(24,16), gridspec_kw=gs_kwarg)
    axes[0].get_shared_x_axes().join(axes[0], *axes)

    axes[0].plot(T, sup, c=SUP_COLOUR, lw=12, alpha=0.5)
    lif_line, = axes[0].plot(T, x_hat_LIF.T, c='k', lw=1)
    axes[0].plot(T, x_hat_Rate.T, ls=':', lw=5, c=lif_line.get_color(), alpha=1)
    axes[0].axis('off')

    for n, idx in enumerate(neuron_idx):
        lif_line, = axes[n+1].plot(T, R_LIF[:,idx], lw=0.5, alpha=1)
        axes[n+1].plot(T, R_Rate[:,idx], ls=':', lw=5, c=lif_line.get_color(), alpha=0.6)
        axes[n+1].axis('off')

    for ax in axes[1:]:
        ax.axvline(x=DT*warmup_steps, ymin=-0.2, ymax=1.2, color='r', label='axvline - full height', clip_on=False)
        ax.axvline(x=DT*(warmup_steps+train_steps), ymin=-0.2, ymax=1.2, color='r', label='axvline - full height', clip_on=False)

    spike_line, = axes[-2].plot(T_phi, d_phi_norm_LIF, c='g', lw=1)
    axes[-2].plot(T_phi, d_phi_norm_LIF_Rate, ls='dotted', lw=3, c=spike_line.get_color(), alpha=0.4)
    axes[-2].axis('off')

    axes[-1].text(0,-0.2, '100 ms', fontsize=SCALE_FONT_SIZE)
    axes[-1].plot([0,0.1], [0,0], lw=5, c='k')
    axes[-1].axis('off')

    # Save data
    if save_dir:
        plt.savefig(f'{save_dir}/fig3_sin.png')
    plt.show()

def plot_frame(data: Data, save_file: str=None, neuron_idx: list=None):
    '''Plots basic panel of data, e.g. pitchfork, oscillator, and Ode to Joy'''

    #### Process Data ####
    R_LIF_test = data.LIF_test['R']
    R_Rate_test = data.Rate_test['R']
    if neuron_idx:
        num_neuron = len(neuron_idx)
    else:
        if not R_LIF_test.shape == R_Rate_test.shape:
            raise ValueError('LIF-spike and LIF-rate neuron time series are not the same shape.')
        num_neuron = R_Rate_test.shape[1]
        neuron_idx = np.arange(num_neuron)

    x_hat_LIF = data.LIF_test['x_hat']
    x_hat_Rate = data.Rate_test['x_hat']
    if not x_hat_LIF.shape == x_hat_Rate.shape:
        raise ValueError('LIF and rate output time series are not the same shape.')
    full_steps = x_hat_Rate.shape[0]
    dim = 1 if x_hat_LIF.ndim == 1 else x_hat_LIF.shape[1]
    T = np.arange(0, DT * full_steps, DT)

    #### Plotting ####
    _, axs = plt.subplots(nrows=num_neuron+dim+1, ncols=1, figsize=(16, num_neuron+dim), sharex=True)

    if dim == 1:
        LIF_line, = axs[0].plot(T, x_hat_LIF, lw=0.5, alpha=0.6)
        axs[0].plot(T, x_hat_Rate, c=LIF_line.get_color(), ls= ':', lw=5, alpha=1)
        axs[0].axis('off')
    else:
        for n in range(dim):
            LIF_line, = axs[n].plot(T, x_hat_LIF[:,n], lw=0.5, alpha=0.6)
            axs[n].plot(T, x_hat_Rate[:,n], c=LIF_line.get_color(), ls= ':', lw=5, alpha=1)
            axs[n].axis('off')

    for n in range(num_neuron):
        LIF_line, = axs[n+dim].plot(T, R_LIF_test[:,n], lw=0.5, alpha=0.6)
        axs[n+dim].plot(T, R_Rate_test[:,n], ls=':', lw=5, c=LIF_line.get_color(), alpha=1)
        axs[n+dim].axis('off')

    axs[-1].text(0, -0.2, '500 ms', fontsize=SCALE_FONT_SIZE)
    axs[-1].plot([0.0,0.5], [0,0], lw=5, c='k')
    axs[-1].axis('off')

    # Save data
    if save_file:
        plt.savefig(save_file)
    plt.show()

def tent_map(z: np.ndarray, window_size: int=21, order: int=1000) -> np.ndarray:
    '''Compute tent map values of given time series.'''

    window = np.ones(window_size) / window_size
    z_smooth = np.convolve(z, window, mode='same')
    z_idx = argrelextrema(z_smooth, np.greater, order=order)
    z_max = z_smooth[z_idx].flatten()
    return z_max

def lorenz_plot(data: Data, save_dir: str=None, neuron_idx: list=None):
    '''Plots Lorenz panel data.'''

    #### Process Data ####
    R_LIF = data.LIF_test['R']
    R_Rate = data.Rate_test['R']
    if neuron_idx:
        num_neuron = len(neuron_idx)
    else:
        if not R_LIF.shape == R_Rate.shape:
            raise ValueError('LIF-spike and LIF-rate neuron time series are not the same shape.')
        num_neuron = R_Rate.shape[1]
        neuron_idx = np.arange(num_neuron)

    x_hat_LIF = data.LIF_test['x_hat']
    x_hat_Rate = data.Rate_test['x_hat']
    if not x_hat_LIF.shape == x_hat_Rate.shape:
        raise ValueError('LIF and rate output time series are not the same shape.')
    full_steps = x_hat_Rate.shape[0]
    dim = 1 if x_hat_LIF.ndim == 1 else x_hat_LIF.shape[1]
    T = np.arange(0, DT * full_steps, DT)

    tent_LIF = tent_map(x_hat_LIF[1000:,2])
    tent_Rate = tent_map(x_hat_Rate[1000:,2])

    #### Plotting Output Planes + Tent ####
    axis_pairs = ((0,2), (0,1), (1,2))
    fig, axes = plt.subplots(ncols=len(axis_pairs)+1, nrows=2, figsize=(18,12))
    for n, pair in enumerate(axis_pairs):
        axes[0,n].plot(x_hat_LIF[::100,pair[0]], x_hat_LIF[::100,pair[1]])
        axes[0,n].axis('off')
        axes[1,n].plot(x_hat_Rate[::100,pair[0]], x_hat_Rate[::100,pair[1]])
        axes[1,n].axis('off')

    axes[0,3].scatter(tent_LIF[:-1], tent_LIF[1:], zorder=2)
    axes[0,3].axis('off')
    axes[1,3].scatter(tent_Rate[:-1], tent_Rate[1:], zorder=2)
    axes[1,3].axis('off')

    # Save data
    if save_dir:
        plt.savefig(f'{save_dir}/fig3_Lorenz_a.png')
    plt.show()

    #### Plotting Neuron Samples####
    _, axes = plt.subplots(nrows=num_neuron+1, ncols=2, figsize=(32,16), sharex=True)

    for n, idx in enumerate(neuron_idx):
        axes[n,0].plot(T, R_LIF[:,idx])
        axes[n,0].axis('off')
        axes[n,1].plot(T, R_Rate[:,idx])
        axes[n,1].axis('off')

    axes[-1,0].text(0, 0, '5 s', fontsize=SCALE_FONT_SIZE)
    axes[-1,0].plot([T[0], T[int(5 / DT)]], [0,0], lw=5, c='k')
    axes[-1,0].axis('off')
    axes[-1,1].text(0, 0, '5 s', fontsize=SCALE_FONT_SIZE)
    axes[-1,1].plot([T[0], T[int(5 / DT)]], [0,0], lw=5, c='k')
    axes[-1,1].axis('off')

    # Save data
    if save_dir:
        plt.savefig(f'{save_dir}/fig3_Lorenz_b.svg')
    plt.show()

def _make_panel(gen_func: callable, plot_func: callable, load_file: str=None, save_file: str=None):
    '''Handles loading, generating and plotting of data.'''

    # Try to load data if load file is given
    if load_file:
        try:
            data = load_date(load_file)
        except FileNotFoundError as e:
            # If file can't be loaded return without throwing exception so other plots can be made
            print('Bad directory for loading data')
            print(e)
            return
    # Generate data if no load file is given
    else:
        data = gen_func()

    # Save data if save file is given
    if save_file:
        save_data(data, save_file)

    # Plot data
    plot_func(data)


## Main function for running as script ##

def main(args):
    if args.panel == 'all':
        _make_panel(sin_data_gen,
            lambda data: sin_plot(data, args.fig_dir),
            f'{args.load_dir}sin.pkl',
            f'{args.save_dir}sin.pkl'
        )
        _make_panel(pitchfork_data_gen,
            lambda data: plot_frame(data, f'{args.fig_dir}pitchfork.png'),
            f'{args.load_dir}pitchfork.pkl',
            f'{args.save_dir}pitchfork.pkl'
        )
        _make_panel(oscillator_data_gen,
            lambda data: plot_frame(data, f'{args.fig_dir}oscillator.png'),
            f'{args.load_dir}oscillator.pkl',
            f'{args.save_dir}oscillator.pkl'
        )
        _make_panel(ode_data_gen,
            lambda data: plot_frame(data, f'{args.fig_dir}ode.png'),
            f'{args.load_dir}ode.pkl',
            f'{args.save_dir}ode.pkl'
        )
        _make_panel(lorenz_data_gen,
            lambda data: lorenz_plot(data, args.fig_dir),
            f'{args.load_dir}lorenz.pkl',
            f'{args.save_dir}lorenz.pkl'
        )
    elif args.panel == 'sin':
        _make_panel(sin_data_gen,
            lambda data: sin_plot(data, args.fig_dir),
            f'{args.load_dir}sin.pkl',
            f'{args.save_dir}sin.pkl'
        )
    elif args.panel == 'pitchfork':
        _make_panel(pitchfork_data_gen,
            lambda data: plot_frame(data, f'{args.fig_dir}pitchfork.png'),
            f'{args.load_dir}pitchfork.pkl',
            f'{args.save_dir}pitchfork.pkl'
        )
    elif args.panel == 'oscillator':
        _make_panel(oscillator_data_gen,
            lambda data: plot_frame(data, f'{args.fig_dir}oscillator.png'),
            f'{args.load_dir}oscillator.pkl',
            f'{args.save_dir}oscillator.pkl'
        )
    elif args.panel == 'ode':
        _make_panel(ode_data_gen,
            lambda data: plot_frame(data, f'{args.fig_dir}ode.png'),
            f'{args.load_dir}ode.pkl',
            f'{args.save_dir}ode.pkl'
        )
    elif args.panel == 'lorenz':
        _make_panel(lorenz_data_gen,
            lambda data: lorenz_plot(data, args.fig_dir),
            f'{args.load_dir}lorenz.pkl',
            f'{args.save_dir}lorenz.pkl'
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            'Makes plots for fig 3 using either newly generated data or loaded from '
            'given directory [load_dir]. If given the figure panels will be saved '
            'to [fig_dir] and if given the data will be saved to [save_dir]. '
            'Which panel is generated can also be specified with [panel].'
        )
    )

    parser.add_argument('--save_dir', type=str, help='Location to save results to')
    parser.add_argument('--load_dir', type=str, help='Location to load results from')
    parser.add_argument('--fig_dir', type=str, help='Location to save figures to')
    parser.add_argument('--panel', type=str, default='all',
        choices=['sin','pithcfork','ode','oscillator', 'lorenz', 'all'],
        help='Which supervisor to generate and/or plot'
    )

    args = parser.parse_args()
    main(args)