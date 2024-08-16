'''Functions used to create figures from generated data.'''

import pickle
import json
from collections import namedtuple
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr as pearsonr
from scipy.stats import linregress as linregress

from pysitronics import networks as nn
from pysitronics import optimization as opt
import supervisors as sups

CORRERlATOIN_LEVELS = np.linspace(0, 1, 100)
ERROR_LEVELS_LIF = np.linspace(-6, -3 , 100)
ERROR_LEVELS_RATE = np.linspace(-6, -3, 100)
ERROR_CUTOFF = 5e-6
SCALE_FONT_SIZE = 28
BIASVAR_LEVELS = np.linspace(-5,1,100)
PROP_LEVELS = np.linspace(0,1,100)

QG_Data = namedtuple('QG_Data', ['err_mat', 'phi', 'QQ', 'GG', 'omega_seed', 'eta_seed', 'psi_seed'])
Entry = namedtuple('Entry', ['g','q','omega_seed','eta_seed','psi_seed'])

#### Data Loading ####
def get_QG_data(supervisor: str, alpha: float, N: int, net_type: str,
                load_dir: str, rep: int=1) -> Tuple[QG_Data, dict, pd.DataFrame]:
    '''Loads data for QG grid sweep from file.

    Uses the given parameters to load the data associated with a QG parameter
    grid sweep then format and return the relavent data.

    Args:
        supervisor: Name of supervisor for desired data
        alpha: Learning rate for desired data
        N: Network size for desired data
        net_type: Type of network for desired data
        load_dir: Location to look for data
        rep: The repetition of the data to load

    Returns:
        data: Loaded data
        config: Configuration parameters of data
        df: Dataframe containing full set of data
    '''

    config_file = f'{load_dir}{supervisor}_{alpha:.1e}_{N}_{net_type}_config_{rep}.json'
    with open(config_file) as f:
        config = json.load(f)

    G_num = config['G_num']
    Q_num = config['Q_num']

    data_file = f'{load_dir}{supervisor}_{alpha:.1e}_{N}_{net_type}_data_{rep}.pkl'
    df = pd.read_pickle(data_file)
    df = df.sort_values(['g','q'])
    err_mat = df['tst_err'].to_numpy().reshape((G_num, Q_num))
    phi = df['phi'].to_numpy().reshape((G_num, Q_num))
    phi = np.array([np.stack(row) for row in phi])

    GG = df['g'].to_numpy().reshape((G_num, Q_num))
    QQ = df['q'].to_numpy().reshape((G_num, Q_num))

    omega_seed = df['omega_seed'].to_numpy(dtype=int).reshape((G_num, Q_num))
    eta_seed = df['eta_seed'].to_numpy(dtype=int).reshape((G_num, Q_num))
    psi_seed = df['psi_seed'].to_numpy(dtype=int).reshape((G_num, Q_num))

    return QG_Data(err_mat, phi, QQ, GG, omega_seed, eta_seed, psi_seed), config_file, df


#### Data Processing ####
def pearson_tensor(A, B):
    '''Computes Pearson correlation pointwise between two matrices of data series.'''

    if not A.shape == B.shape:
        raise ValueError('Matrices must be the same shape.')

    pearc_mat = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            pearc_mat[i,j] = pearsonr(A[i,j].flatten(), B[i,j].flatten())[0]

    return pearc_mat

def test_network(phi: np.ndarray, g: float, q: float, omega_seed: int, eta_seed: int,
                 psi_seed: int, state: dict, config: dict, saves: dict={}) -> Tuple[np.ndarray, dict]:
    '''Creates network with given parameters and simulates it.

    Args:
        phi: Decoder
        g: Reservoir strength
        q: Encoder strength
        omega_seed: Random seed for reservoir
        eta_seed: Random seed for encoder
        psi_seed: Random seed for input weight matric
        state: State to load into network
        config: Configuration of simulation to load
        saves: Values to save for network

    Returns:
        x_hat_save: Timeseries of network output
        saves: Dictionary of timeseries for desired values
    '''

    omega = nn.sigma_gen(config['N'], g, config['p'], omega_seed)
    if config['row_balance']:
        nn.row_balance(omega)
    eta = nn.omega_gen(config['N'], q, eta_seed, config['dim'])
    psi = nn.omega_gen(config['N'], q, psi_seed, config['dim'])

    if config['net_type'] not in ['LIF', 'LIF_Rate']:
        raise ValueError('Given network class is not valid.')

    if config['net_type'] == 'LIF':
        net_class = nn.LIF
    elif config['net_type'] == 'LIF_Rate':
        net_class = nn.LIF_Rate

    net = nn.net_class(
        omega = omega,
        eta = eta,
        psi = psi,
        t_m = config['t_m'],
        t_ref = config['t_ref'],
        v_reset = config['v_reset'],
        v_peak = config['v_peak'],
        i_bias = config['i_bias'],
        t_r = config['t_r'],
        t_d = config['t_d'],
        dim = config['dim'],
    )

    net.set_state(state)
    net.set_decoder(phi)

    sup, I_in = sups.get_supervisor(config['superverisor'], config['dt'])

    return net.simulate(config['dt'], sup.shape[0] * config['test_epochs'], I_in= I_in, save_values=saves)

def align_xhat_sin(x_hat: np.ndarray, config: dict, window_size: int=501,
                   smooth: bool=False) -> Tuple[np.ndarray,np.ndarray]:
    '''Time aligns a continous time series of outputs into a vector of outputs accounting for drift.

    Arranges a single timeseries vector of outputs into an array out outputs and
    time aligns each individual timeseries by finding the maximum of output and
    aligning it to the time the supervisor has its maximum. If smooth==True then
    the timeseries are smoothed before finding the maximum. This is needed in the
    case of the LIF-spiking outputs due to the inherent non-smooth outputs resulting
    in many local extrema.

    Args:
        x_hat: Timeseries vector
        config: Configuration of simulations
        window_size: Size of smoothing window used for finding time alignment
        smooth: Whether to smooth before finding maximums

    Returns:
        x_hat_reorder: Resulting vector of timeseries
        x_hat_idx: Index of maximum for each timeseries
    '''

    if not config['supervisor'].split('_')[0] == 'sin':
        raise ValueError('Time aligning assumes supervisor is sin.')

    # Load QG and supervisor details from config
    QG_grid_size = config['Q_num'] * config['G_num']
    sup = sups.get_supervisor(config['supervisor'], config['sample_rate'] * config['dt'])
    offset = np.argmax(sup)

    # Reshape outputs into matrix of outputs
    x_hat = x_hat.reshape(QG_grid_size, sup.size,-1,order='F')
    # Smooth network output if required
    if smooth:
        window = np.ones(window_size) / window_size
        x_hat_smooth = np.apply_along_axis(lambda m: np.convolve(m, window, mode='same'), axis=1, arr=x_hat)
        # Reshape continous smoothed output into matrix
        x_hat_smooth = x_hat_smooth.reshape(QG_grid_size, sup.size,-1,order='F')
        # Use smoothed outputs to compute index of maximum for each rep
        x_hat_idx = np.argmax(x_hat_smooth, axis=1)
    else:
        # Use unsmoothed network output to compute index of maximum for each rep
        x_hat_idx = np.argmax(x_hat, axis=1)

    # Use max indeces to reorder each rep into new vector
    x_hat_reorder = np.zeros((QG_grid_size, sup.size, config['test_epochs']))
    for n,row in enumerate(x_hat):
        for m,x in enumerate(row.T):
            x_hat_reorder[n,:,m] = np.roll(x, offset - x_hat_idx[n,m])

    return x_hat_reorder, x_hat_idx

def compute_bias_variance_prop(x_hat: np.ndarray, x: np.ndarray) -> Tuple[float, float, float]:
    '''Computes average bias squared, variance and their proportion of a matrix of timeseries.

    Args:
        x_hat: Matrix of approximant timeseries
        x: Target timeseries

    Returns:
        bias_squared: The time averaged bias squared
        var: The time averaged variance
        prop: The proporiton of variance to the sum of the bias squared variance
    '''

    bias_squared = np.power(x_hat.mean(-1) - x,2).mean(-1)
    var = x_hat.var(-1).mean(-1)
    prop = var / (bias_squared + var)

    return bias_squared, var, prop

#### Plotting ####
def plot_frame(x_hat: np.ndarray, R: np.ndarray, config:dict, save_file: str=None,
               neuron_idx: list=None, auto_close: bool=True) -> None:
    '''Plots data frame of a networks output and sample neuron dynamics.

    Args:
        x_hat: Network output data
        R: Neuron dynamics
        config: Configuration of simulation
        save_file: File to save figure to
        neuron_idx: List of idices forsample neuron dynamics to plot
        auto_close: Whether or not to close figure window after saving
    '''

    #### Process Data ####
    if neuron_idx:
        num_neuron = len(neuron_idx)
    else:
        num_neuron = R.shape[1]
        neuron_idx = np.arange(num_neuron)

    full_steps = x_hat.shape[0]
    dim = 1 if x_hat.ndim == 1 else x_hat.shape[1]
    T = np.arange(0, config['dt'] * full_steps, config['dt'])

    #### Plotting ####
    _, axs = plt.subplots(nrows=num_neuron+dim+1, ncols=1, figsize=(16, num_neuron+dim), sharex=True)

    if dim == 1:
        axs[0].plot(T, x_hat)
        axs[0].axis('off')
    else:
        for n in range(dim):
            axs[n].plot(T, x_hat[:,n])
            axs[n].axis('off')

    for n in range(num_neuron):
        axs[n+dim].plot(T, R[:,n])
        axs[n+dim].axis('off')

    axs[-1].text(0, -0.2, '500 ms', fontsize=SCALE_FONT_SIZE)
    axs[-1].plot([0.0,0.5], [0,0], lw=5, c='k')
    axs[-1].axis('off')

    # Save data
    if save_file:
        plt.savefig(save_file)
        if auto_close:
            plt.close()
            return
    plt.show()

def fig_4_5_panel(supervisor: str, alpha: float, N: int, load_dir: str, rep: int=1,
                  save_dir: str=None, auto_close: bool=True) -> None:
    '''Plots panel of figures 4 and 5 for a given supervisor and trained network.

    Args:
        supervisor: Name of supervisor
        alpha: Learning rate of simulation
        N: Network size
        load_dir: Directory to load data from
        rep: Repretition of datas
        save_dir: Directory to save figure to
        auto_close: Whether or not to close figure window after saving
    '''

    save_file = f'{save_dir}{supervisor}_{alpha:.1e}_{N}_{rep}'

    ## Get data ##
    LIF_data, LIF_config, LIF_df = get_QG_data(supervisor, alpha, N, 'LIF', load_dir, rep)
    Rate_data, Rate_config, Rate_df = get_QG_data(supervisor, alpha, N, 'LIF_Rate', load_dir, rep)
    corr_mat = pearson_tensor(LIF_data.phi, Rate_data.phi)


    if not np.allclose(LIF_data.GG, Rate_data.GG):
        raise ValueError('QG grind is not the same for LIF-spike and LIF-rate.')
    if not np.allclose(LIF_data.QQ, Rate_data.QQ):
        raise ValueError('QG grind is not zthe same for LIF and LIF-rate.')
    if not np.all(LIF_data.omega_seed == Rate_data.omega_seed):
        raise ValueError('Omega seeds are not the same for LIF and LIF-rate.')
    if not np.all(LIF_data.eta_seed == Rate_data.eta_seed):
        raise ValueError('Eta seeds are not the same for LIF and LIF-rate.')
    if not np.all(LIF_data.psi_seed == Rate_data.psi_seed):
        raise ValueError('Psi seeds are not the same for LIF and LIF-rate.')

    QQ, GG = LIF_data.QQ, LIF_data.GG

    ## Plot QG heat maps ##
    qg_corr_panel(QQ, GG, LIF_data.err_mat, Rate_data.err_mat, corr_mat, f'{save_file}_heatMaps.png', auto_close=auto_close)

    ## Perform decoder swap ##
    # Find (Q,G) point to swap
    LIF_mask = LIF_data.err_mat < ERROR_CUTOFF
    Rate_mask = Rate_data.err_mat < ERROR_CUTOFF
    most_corr = float('-inf')
    most_corr_idx_QG = (0,0)
    for n, row in enumerate(LIF_mask):
        for m, thres_flag in enumerate(row):
            if LIF_mask[n,m] and Rate_mask[n,m]:
                if most_corr < corr_mat[n,m]:
                    most_corr = corr_mat[n,m]
                    most_corr_idx_QG = (n,m)
    most_corr_idx = 40 * most_corr_idx_QG[0] + most_corr_idx_QG[1]

    # Run simulations
    saves = {'R':10}
    LIF_x_hat, LIF_saves = test_network(
        LIF_data.phi[most_corr_idx_QG],
        LIF_df.iloc[most_corr_idx]['g'],
        LIF_df.iloc[most_corr_idx]['q'],
        LIF_df.iloc[most_corr_idx]['omega_seed'],
        LIF_df.iloc[most_corr_idx]['eta_seed'],
        LIF_df.iloc[most_corr_idx]['psi_seed'],
        LIF_df.iloc[most_corr_idx]['state'],
        LIF_config,
        saves
    )
    Rate_x_hat, Rate_saves = test_network(
        Rate_data.phi[most_corr_idx_QG],
        Rate_df.iloc[most_corr_idx]['g'],
        Rate_df.iloc[most_corr_idx]['q'],
        Rate_df.iloc[most_corr_idx]['omega_seed'],
        Rate_df.iloc[most_corr_idx]['eta_seed'],
        Rate_df.iloc[most_corr_idx]['psi_seed'],
        Rate_df.iloc[most_corr_idx]['state'],
        Rate_config,
        saves
    )
    # Swapped decoders
    LIF_x_hat_swap, LIF_saves_swap = test_network(
        Rate_data.phi[most_corr_idx_QG],
        LIF_df.iloc[most_corr_idx]['g'],
        LIF_df.iloc[most_corr_idx]['q'],
        LIF_df.iloc[most_corr_idx]['omega_seed'],
        LIF_df.iloc[most_corr_idx]['eta_seed'],
        LIF_df.iloc[most_corr_idx]['psi_seed'],
        LIF_df.iloc[most_corr_idx]['state'],
        LIF_config,
        saves
    )
    Rate_x_hat_swap, Rate_saves_swap = test_network(
        LIF_data.phi[most_corr_idx_QG],
        Rate_df.iloc[most_corr_idx]['g'],
        Rate_df.iloc[most_corr_idx]['q'],
        Rate_df.iloc[most_corr_idx]['omega_seed'],
        Rate_df.iloc[most_corr_idx]['eta_seed'],
        Rate_df.iloc[most_corr_idx]['psi_seed'],
        Rate_df.iloc[most_corr_idx]['state'],
        Rate_config,
        saves
    )

    ## Plot swapped decoder ##
    plot_frame(LIF_x_hat, LIF_saves['R'], LIF_config, f'{save_file}_LIF.png', auto_close=auto_close)
    plot_frame(Rate_x_hat, Rate_saves['R'], Rate_config, f'{save_file}_Rate.png', auto_close=auto_close)
    plot_frame(LIF_x_hat_swap, LIF_saves_swap['R'], LIF_config, f'{save_file}_LIF_Swap.png', auto_close=auto_close)
    plot_frame(Rate_x_hat_swap, Rate_saves_swap['R'], Rate_config, f'{save_file}_Rate_Swap.png', auto_close=auto_close)

    ## Plot decoder linear fit ##
    slope, intercept, r, p, stderr = linregress(Rate_data.phi[most_corr_idx_QG].flatten(), LIF_data.phi[most_corr_idx_QG].flatten())
    fit_text = rf'$\phi_S$={slope:.2f}$\phi_R$ + {intercept:.2e}, r={r:.2f}'

    _, axs = plt.subplots(ncols=2, figsize=(4,6))
    axs[0].scatter(Rate_data.phi[most_corr_idx_QG], LIF_data.phi[most_corr_idx_QG], color='g', s=10)
    axs[0].plot(Rate_data.phi[most_corr_idx_QG], intercept + slope * LIF_data.phi[most_corr_idx_QG], lw=5, c='k')
    axs[0].set_box_aspect(1)
    axs[0].axis('off')
    axs[1].text(5,14, fit_text, fontsize=SCALE_FONT_SIZE)
    axs[1].axis('off')

    # Save data
    if save_file:
        plt.savefig(f'{save_file}_linearfit.png')
        if auto_close:
            plt.close()
            return
    plt.show()

def qg_corr_panel(QQ, GG, LIF_err_mat, Rate_err_mat, corr_mat, save_file:
                  str=None, auto_close: bool=True) -> None:
    '''Plots QG grid sweep heatmaps of LIF-spiking error, LIF-rate error, and cross network decoder correlation.

    Args:
        QQ: Numpy meshgrid of Q points
        GG: Numpy meshgrid of G points
        LIF_err_mat: Matrix of LIF-spiking testing error
        Rate_err_mat: Matrix of LIF-rate testing error
        corr_mat: Correlation matrix between learned LIF-spiking and LIF-rate decoders
        save_file: File to save figure to
        auto_close: Whether or not to close figure window after saving
    '''

    ## Plot ##
    _, axes_cont = plt.subplots(constrained_layout=True, nrows=3, figsize=(5,12), sharex=True)

    cs_1 = axes_cont[0].contourf(QQ, GG, np.log10(LIF_err_mat), levels=ERROR_LEVELS_LIF, extend='min')
    plt.colorbar(cs_1, ax=axes_cont[0])
    cs_2 = axes_cont[1].contourf(QQ, GG, np.log10(Rate_err_mat), levels=ERROR_LEVELS_RATE, extend='min')
    plt.colorbar(cs_2, ax=axes_cont[1])
    cs_3 = axes_cont[2].contourf(QQ, GG, corr_mat, levels=CORRERlATOIN_LEVELS, extend='min')
    plt.colorbar(cs_3, ax=axes_cont[2])

    # Square Plot Ratios
    for ax in axes_cont:
        ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.tick_params(bottom=False, left=False)

    # Save and/or show figure
    if save_file:
        plt.savefig(fname = f'{save_file}_contour.png')
        if auto_close:
            plt.close()
            return
    plt.show()

def bias_variance(supervisor: str, alpha: float, N: int, net_type: str,
        load_dir: str, rep: int=1, save_file: str=None, auto_close: bool=True) -> None:
    '''Plots bias-variance decomposition plots (fig 7) for given data.

    Args:
        supervisor: Name of supervisor
        alpha: Learning rate of simulation
        N: Network size
        net_type: Type of network
        load_dir: Directory to load data from
        rep: Repretition of datas
        save_dir: Directory to save figure to
        auto_close: Whether or not to close figure window after saving
    '''

    ## Load data and config ##
    load_path = f'{load_dir}{supervisor}_{alpha:.1e}_{N}_{rep}'
    with open(f'{load_path}_config_{rep}.json') as f:
        config  = json.load(f)
    df = pd.read_pickle(f'{load_path}_data_{rep}.pkl')
    df.sort_values(['g','q'], inplace=True)
    sup = sups.get_supervisor(config['supervisor'], config['sample_rate'] * config['dt'])
    QG_grid_dim = (config['G_num'], config['G_num'])

    ## Compute bias variance ##
    x_hat = np.array(f['x_hat_test'].to_list())[:,:-1]
    if net_type == 'LIF':
        x_hat_ordered = align_xhat_sin(x_hat, config, smooth=True)
    elif net_type == 'LIF_Rate':
        x_hat_ordered = align_xhat_sin(x_hat, config, smooth=False)

    bias_squared, variance, prop = compute_bias_variance_prop(x_hat_ordered, sup)
    del x_hat_ordered
    bias_squared = bias_squared.reshape(*QG_grid_dim)
    variance = variance.reshape(*QG_grid_dim)
    prop = prop.reshape(*QG_grid_dim)

    ## Plotting ##
    _, axes_cont = plt.subplots(constrained_layout=True, nrows=3, figsize=(5,12), sharex=True)

    cs_1 = axes_cont[0].contourf(np.log10(bias_squared), levels=BIASVAR_LEVELS, extend='both')
    plt.colorbar(cs_1, ax=axes_cont[0])
    cs_2 = axes_cont[1].contourf(np.log10(variance), levels=BIASVAR_LEVELS, extend='both')
    plt.colorbar(cs_2, ax=axes_cont[1])
    cs_3 = axes_cont[2].contourf(prop, levels=PROP_LEVELS, extend='both')
    plt.colorbar(cs_3, ax=axes_cont[2])

    # Square Plot Ratios
    for ax in axes_cont:
        ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.tick_params(bottom=False, left=False)

    # Save and/or show figure
    if save_file:
        plt.savefig(fname = f'{save_file}_contour.png')
        if auto_close:
            plt.close()
            return
    plt.show()



