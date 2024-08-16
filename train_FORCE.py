'''Script for FORCE training a supervisor to networks over QG parameter grid.

Given a specific supervisor, training/testing repetitions, learning rate factor
and network type, will train and test networks over the given QG grid. The default
QG grid has the meshpoints Q = np.linspace(0.2, 30, 40) and G = np.linspace(0, 0.2, 40).
If data with the same parameters has already been saved to the given save directory,
the repetition number will be incremented and appened to the name of the data files
so as to avoid overwritting past data.

Examples:
    The following will generate data for Fig 4, and save the data to the directory
    Data/
    >>> python train_FORCE --supervisor sin_5 --train_epochs 6 --test_epochs 6 --alpha 5e-6 --net_type LIF
    >>> python train_FORCE --supervisor sin_5 --train_epochs 6 --test_epochs 6 --alpha 5e-6 --net_type LIF_Rate
    >>> python train_FORCE --supervisor ode_HDTS --train_epochs 20 --test_epochs 2 --alpha 5e-8 --net_type LIF
    >>> python train_FORCE --supervisor ode_HDTS --train_epochs 20 --test_epochs 2 --alpha 5e-8 --net_type LIF_Rate
    >>> python train_FORCE --supervisor pitchfork --train_epochs 1 --test_epochs 1 --alpha 5e-6 --net_type LIF
    >>> python train_FORCE --supervisor pitchfork --train_epochs 1 --test_epochs 1 --alpha 5e-6 --net_type LIF_Rate
    The following will generate data for Fig 5, and save the data to the directory
    Data/
    >>> python train_FORCE --supervisor sin_5 --train_epochs 6 --test_epochs 6 --alpha 5e-4 --net_type LIF
    >>> python train_FORCE --supervisor sin_5 --train_epochs 6 --test_epochs 6 --alpha 5e-4 --net_type LIF_Rate
    >>> python train_FORCE --supervisor ode_HDTS --train_epochs 20 --test_epochs 2 --alpha 5e-3 --net_type LIF
    >>> python train_FORCE --supervisor ode_HDTS --train_epochs 20 --test_epochs 2 --alpha 5e-3 --net_type LIF_Rate
    >>> python train_FORCE --supervisor pitchfork --train_epochs 1 --test_epochs 1 --alpha 5e0 --net_type LIF
    >>> python train_FORCE --supervisor pitchfork --train_epochs 1 --test_epochs 1 --alpha 5e0 --net_type LIF_Rate
'''

from argparse import Namespace
import os, time
from datetime import date
from collections import namedtuple
import json
from typing import List

import numpy as np
import pandas as pd

from pysitronics import networks as nn
from pysitronics import optimization as opt
import supervisors as sups


Entry = namedtuple('Entry', ['g','q','omega_seed','eta_seed','psi_seed'])

class Job_Starter:
    '''Creates, trains and tests networks.'''

    def __init__(self, args: dict):
        supervisor, I_in = sups.get_supervisor(args['supervisor'], args['dt'])
        sup_length = supervisor.shape[0]
        self.dim = 1 if supervisor.ndim == 1 else supervisor.shape[1]
        self.args = args
        self.optimizer = opt.FORCE(
            args['alpha'],
            supervisor,
            args['dt'],
            args['warmup_steps'],
            args['train_epochs'] * sup_length,
            args['learn_step'],
            I_in
        )
        self.evaluator = opt.Evaluator(
            supervisor = supervisor,
            eval_steps = args['test_epochs'] * sup_length,
            sample_rate = args['sample_rate'],
            dt = args['dt'],
            I_in = I_in
        )

    def run_simulation(self, params: Entry, net_type: str):
        '''Takes Q,G parameters then creates network to be trained and tested on a task.'''

        # Create network connectivity matrices
        omega = nn.omega_gen(self.args['N'], params.g, self.args['p'], params.omega_seed)
        if self.args['row_balance']:
             nn.row_balance(omega)
        eta = nn.eta_gen(self.args['N'], params.q, seed=params.eta_seed, dim=self.dim)
        # Input weight matrix
        psi = nn.eta_gen(self.args['N'], params.q, seed=params.psi_seed, dim=self.dim)
        # Create network
        network_type = nn.LIF if net_type == 'LIF' else nn.LIF_Rate
        net = network_type(
            self.args['N'],
            omega,
            eta,
            self.args['t_m'],
            self.args['t_ref'],
            self.args['v_reset'],
            self.args['v_peak'],
            self.args['i_bias'],
            self.args['t_r'],
            self.args['t_d'],
            self.dim,
            psi
        )

        # Train network
        _ = self.optimizer.teach_network(net)
        # Test network
        results = self.evaluator.test_network(net)
        # Add parameters to results
        results.update(params._asdict())

        return results


def make_sweep(Q_min: float, Q_max: float, Q_num: int, G_min: float,
               G_max: float, G_num: int, start_seed: int=None):
    '''Generates list of Q,G parameters to perform grid sweep.'''

    Q = np.linspace(Q_min, Q_max, Q_num)
    G = np.linspace(G_min, G_max, G_num)
    if start_seed:
        sweep = [Entry(
                g, q,
                omega_seed = start_seed + 3*(n*G_num + m),
                eta_seed = start_seed + 3*(n*G_num + m) + 1,
                psi_seed = start_seed + 3*(n*G_num + m) + 2
            ) for n, g in enumerate(G) for m, q in enumerate(Q)
        ]
    else:
        sweep = [Entry(g, q, None, None, None) for g in G for q in Q]

    return sweep

def save_data(results: List[dict], args: dict):
    '''Saves results and parsed arguments.'''

    # Make path to save files
    data_file = f"{args['save_dir']}{args['supervisor']}_{args['alpha']}_{args['N']}_{args['net_type']}_data"
    config_file = f"{args['save_dir']}{args['supervisor']}_{args['alpha']}_{args['N']}_{args['net_type']}_config"

    # Create data directory if it does not exist
    if not os.path.exists(args['save_dir']):
            print('Directory:', args['save_dir'], '\ndoes not exist, creating new one.')
            os.makedirs(args['save_dir'])

    # Add run number to avoid overwritting previous data
    run_number = 1
    while os.path.exists(f'{data_file}_{run_number}.pkl') or os.path.exists(f'{config_file}_{run_number}.json'):
            run_number += 1

    data_file = f'{data_file}_{run_number}.pkl'
    config_file = f'{config_file}_{run_number}.json'

    print('Saving data to:', data_file)
    print('Saving config to:', config_file)

    # Save configuration
    with open(config_file, 'w') as configfile:
        args['Date'] = str(date.today())
        json.dump(args, configfile, indent=4)

    # Save simulation results
    pd.DataFrame(results).to_pickle(data_file)

    print('Save complete.')


## Main function for running as script ##

def main(args):
    # Start time
    tic = time.perf_counter()

    # Initilize job starter and parameter sweep
    print('Initializing')
    job_starter = Job_Starter(vars(args))
    sweep = make_sweep(args.Q_min, args.Q_max, args.Q_num, args.G_min, args.G_max, args.G_num, args.start_seed)

    # Run simulations
    print('Running simulations')
    # NOTE: It is possible to parellize the following operation using the multiprocessing
    #       module which might offer an advatange on high core machines
    results = [job_starter.run_simulation(params, args.net_type) for params in sweep]

    # Save data
    print('Saving')
    args = vars(args)
    args['dim'] = job_starter.dim
    save_data(results, args)

    # End timer
    toc = time.perf_counter()
    time_dif = toc - tic
    s = time_dif % 60
    m = (time_dif % (60*60)) // 60
    h = (time_dif % (60*60*24)) // (60*60)
    d = time_dif // (60*60*24)

    print(f'Done. Total time: {d}:{h:2.0f}:{m:2.0f}:{s:2.0f}')


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Script for FORCE training a supervisor to networks over QG parameter grid. See docstring for details."
    )

    parser.add_argument('--dt', type=float, default=5e-5, help='Integration time step (s)')
    # Training Parameters
    parser.add_argument('--alpha', type=float, required=True, default=SUPPRESS,
                            help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='Number of warmup steps before training')
    parser.add_argument('--learn_step', type=int, default=100, help='Number of steps between each RLS update')
    parser.add_argument('--supervisor', required=True, default=SUPPRESS,
                            help='Name of supervisors to train network')
    parser.add_argument('--train_epochs', type=int, required=True, default=SUPPRESS,
                            help='Number of repetitions of supervisor used in training')
    # Testing Parameters
    parser.add_argument('--test_epochs', type=int, required=True, default=SUPPRESS,
                            help='Number of repetitions of supervisor used in testing')
    parser.add_argument('--sample_rate', type=int, default=100,
                            help='Number of steps between saved network output')
    parser.add_argument('--save_dir', default='Data/', help='Location to save data to')
    # Neuron Parameters
    parser.add_argument('--t_m', type=float, default=1e-2, help='Membrane time constant (s)')
    parser.add_argument('--t_ref', type=float, default=2e-3, help='Refractory time (s)')
    parser.add_argument('--v_reset', type=float, default=-65, help='Reset voltage (mV)')
    parser.add_argument('--v_peak', type=float, default=-40, help='Peak voltage (mV)')
    parser.add_argument('--i_bias', type=float, default=-40, help='Bias current (mV)')
    parser.add_argument('--t_r', type=float, default=2e-3, help='Rise time (s)')
    parser.add_argument('--t_d', type=float, default=0.02, help='Decay time (s)')
    # Network Parameters
    parser.add_argument('--net_type', required=True, choices=['LIF','LIF_Rate'], default=SUPPRESS,
                            help='Type of neuron to use')
    parser.add_argument('-N', '--net_size', dest='N', default=2000, type=int,
                            help='Number of neurons in network')
    parser.add_argument('-p', '--sparsity', dest='p', default=0.4, type=float,
                            help='Sparsity of reservoir connections')
    parser.add_argument('--row_balance', type=bool, default=True, help='Is omega^0 row balanced?')
    parser.add_argument('--Q_min', type=float, default=2, help='Lowest Q value used in grid')
    parser.add_argument('--Q_max', type=float, default=30, help='Highest Q value used in grid')
    parser.add_argument('--Q_num', type=int, default=40, help='Number of Q values used in grid')
    parser.add_argument('--G_min', type=float, default=0, help='Lowest G value used in grid')
    parser.add_argument('--G_max', type=float, default=.2, help='Highest G value used in grid')
    parser.add_argument('--G_num', type=int, default=40, help='Number of G values used in grid')
    parser.add_argument('--start_seed', type=int, help='First seed used in random number generator for grid')

    args = parser.parse_args()
    main(args)