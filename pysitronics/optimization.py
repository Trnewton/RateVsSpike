from collections import namedtuple
from typing import Tuple
import numpy as np

from pysitronics import networks as nn


#### Functions ####
def L2_error(x: np.ndarray, y: np.ndarray, dt: float):
    '''Computes time averaged L2 error between x and y.'''
    return np.sqrt(np.sum((x - y) * (x - y)) * dt) / x.shape[0]


#### Classes ####
class FORCE:
    '''Trains a network using the FORCE technique.

    Attributes:
        alpha (float): Learning rate
        sup (np.ndarray): Target dynamics
        dt (float): Integration time step
        warmup_steps (int): Number of integration steps before training begins
        training_steps (int): Number of steps to train network for
        learn_step (int): Number of integration steps between FORCE updates
        I_in (np.ndarray): Network input for task
    '''

    def __init__(self,
            alpha: float,
            supervisor: np.ndarray,
            dt: float,
            warmup_steps: int,
            training_steps: int,
            learn_step: int,
            I_in: np.ndarray=None
        ):

        self.alpha = alpha # Learning rate
        self.sup = supervisor # Target dynamics
        self.dt = dt # Integration time step
        self.warmup_steps = warmup_steps # Pre-training warmup steps
        self.training_steps = training_steps # Number of steps to traing for
        self.learn_step = learn_step # Number of steps between decoder updates
        self.I_in = I_in # Input current to network

    def teach_network(self, network: nn.Abstract_Network, save_values: dict={'x_hat':None}) -> Tuple[dict, dict]:
        '''Trains given network with FORCE on target supervisor.

        Takes a network and trains it with the initilized FORCE training parameters
        to perform the prespecified task. The attributes of the network named in save_values
        are saved at each time step and returned as numpy ndarrays in a dictionary.
        The dict save_values should consist of keys (str) indicating the name of the
        attribute to save and a value (int/None) to indicate how many neuron samples to save.
        If the given value is an int then that number of neuron samples will be saved
        and if it is None then parameter for all neurons will be saved.

        Args:
            network: The network that will be trained
            save_values: The names of network attributes and number of samples to be saved

        Returns:
            warmup_saves: Saved values from save_values during warmup
            training_saves: Saved values from save_values during training
        '''

        if not all(k in vars(network) for k in save_values.keys()):
            raise ValueError('Save value is not an attribute of network.')

        # PreTraining network run
        _, warmup_saves = network.simulate(self.dt, self.warmup_steps, save_values=save_values)
        training_saves = {
            value : np.zeros((self.training_steps))
                    if num_save is None else
                    np.zeros((self.training_steps, num_save))
                    for value, num_save in save_values.items()
        }

        # Initilize P matrix
        P = self.alpha * np.eye(network.N)
        c = 0
        k = 0
        # Simulate for training length
        for n in range(self.training_steps):
            # Perform intergration step of network
            if self.I_in is not None:
                z, R = network.step(self.dt, self.I_in[n])
            else:
                z, R = network.step(self.dt)

            # Perform update to decoder if on an update step
            if (n % self.learn_step) == 0:
                # Compute error
                signal = self.sup[n%self.sup.shape[0]]
                error = z - signal
                # Update approximation of inverse cross-correlation matrix
                k = np.dot(R, P)
                rPr = np.dot(R, k)
                c = 1.0 / (1.0 + rPr)
                P = P - c * np.outer(k, k)
                # Compute update to decoder
                if network.dim == 1:
                    dphi = - c * error * k
                else:
                    dphi = - c * np.outer(error, k)
                network.set_decoder(network.get_decoder() + dphi)

            # Save network state values if given
            for value, num_save in save_values.items():
                if num_save is None:
                    training_saves[value][n] = getattr(network, value)
                else:
                    training_saves[value][n] = getattr(network, value)[:num_save]

        return  warmup_saves, training_saves

class Evaluator:
    '''Evaluates a trained network on a specified task.

    Attributes:
        sup (np.ndarray): Target dynamics
        eval_steps (int): Number of steps to evaluate network for
        sample_rate (int): Number of integration steps saved samples
        dt (float): Integration step time
        I_in (np.ndarray): Network input for task
    '''

    def __init__(self,
            supervisor: np.ndarray,
            eval_steps: int,
            sample_rate: int,
            dt: float,
            I_in: np.ndarray
        ):

        self.sup = supervisor
        self.I_in = I_in
        self.eval_steps = eval_steps
        self.sample_rate = sample_rate
        self.dt = dt
        self.samples = self.eval_steps // self.sample_rate
        self.dim = self.sup.shape[1] if self.sup.ndim == 2 else 1

    def test_network(self, network: nn.Abstract_Network, save_values:dict={}) -> dict:
        '''Tests given network on task.

        Takes a network and trains it with the initilized FORCE training parameters
        to perform the prespecified task. The attributes of the network named in save_values
        are saved at each time step and returned as numpy ndarrays in a dictionary.
        The dict save_values should consist of keys (str) indicating the name of the
        attribute to save and a value (int/None) to indicate how many neuron samples to save.
        If the given value is an int then that number of neuron samples will be saved
        and if it is None then parameter for all neurons will be saved. In addition
        the testing error, network state, learned decoder and testing output are
        saved with the keys 'tst_err', 'state', 'phi', and 'x_hat_test'.

        Args:
            network: The network that will be trained
            save_values: The names of network attributes and number of samples to be saved

        Returns:
            saves: Saved values from save_values plus error, state, and decoder
        '''

        if not all(k in vars(network) for k in save_values.keys()):
            raise ValueError('Given save value is not an attribute of network.')

        # Save structures
        saves = {value:np.zeros((self.samples, num_save)) for value, num_save in save_values.items()}
        saves['state'] = network.get_state()
        saves['phi'] = network.get_decoder()
        saves['x_hat_test'] = np.zeros((self.samples, self.dim)).squeeze()
        test_error = 0

        for n in range(self.eval_steps):
            # Perform integration step
            if self.I_in is not None: # External input current to network
                z, R = network.step(self.dt, self.I_in[n])
            else: # No external input current to network
                z, R = network.step(self.dt)
            # Store testing data
            if (n % self.sample_rate) == 0:
                saves['x_hat_test'][int(n/self.sample_rate)] = z
                for value, num_save in save_values.items():
                    saves[value][int(n/self.sample_rate)] = getattr(network, value)[:num_save]
            # Get value of supervisor and compute error
            f = self.sup[n % self.sup.shape[0]]
            test_error += (f-z) * (f-z)

        # Compute normalized L2 error across supervisor dimensions, integration time step, integration time
        test_error = np.sqrt(np.sum(test_error) * self.dt) / self.eval_steps
        if self.dim != 1:
            test_error = np.mean(test_error)
        # Save testing error
        saves['tst_err'] = test_error

        return saves