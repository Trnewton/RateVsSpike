from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


#### Functions ####
def omega_gen(N: int, g: float, p: float, seed: int=None) -> np.ndarray:
    '''Generates reservoir weight matrix of networks.

    Args:
        N: Network size
        g: Connection strength
        p: Connection sparsity
        seed: Random seed used
    '''

    if seed is not None:
        np.random.seed(seed)

    return g * (np.random.randn(N, N)) * (np.random.rand(N, N) < p) / (np.sqrt(N) * p)

def row_balance(matrix) -> None:
    '''In place row balances given matrix so that each row sum is zero while retaining any zero weights.'''

    for n, row in enumerate(matrix):
        if np.sum(row) == 0:
            continue
        matrix[n, row.nonzero()] -= np.sum(row) / row.nonzero()[0].size

def eta_gen(N: int, q: float=2.0, shift: float=-0.5, dim: int=1, seed=None) -> np.ndarray:
    '''Generates feedback and input weight matrices of networks.

    Args:
        N: Network size
        q: Connection strength
        shift: Mean of connection strengths
        dim: Dimension of supervisor dynamics
        seed: Random seed used to generate connections

    Returns:
        np.ndarray: Resulting connectivity matrix
    '''

    if seed is not None:
        np.random.seed(seed)

    if dim == 1:
        eta = q * (np.random.rand(N) + shift)
    else:
        eta = q * (np.random.rand(N, dim) + shift)

    return eta


#### Classes ####
class Abstract_Network(ABC):
    '''Template class for networks which indicates which methods must be implemented to work with optimization module.'''

    @abstractmethod
    def step(self, dt: float, i_in: np.ndarray=0) -> Tuple[np.ndarray, dict]:
        '''Intergrates network one time step then returns the output and neural activation of the network.'''

    @abstractmethod
    def simulate(self, dt: float, N: int, I_in: np.ndarray=None, save_values:dict={}) -> Tuple[np.ndarray, dict]:
        '''Simulates network for N steps with time steps of size dt, returning the output.'''

    @abstractmethod
    def get_state(self) -> dict:
        '''Returns enough information about the state of the network to reproduce its dynamics.'''

    @abstractmethod
    def set_state(self, new_state: dict) -> None:
        '''Sets all required state parameters to values from new_state '''

    @abstractmethod
    def get_decoder(self) -> np.ndarray:
        '''Gets value of network decoder.'''

    @abstractmethod
    def set_decoder(self, new_phi: np.ndarray) -> None:
        '''Updates the decocder.'''

class LIF(Abstract_Network):
    '''Simulates spiking based leaky-integrate-and-fire reservoir neural network.

    Attributes:
        t_m (float): Membrane time constant
        t_ref (float): Refractor period length
        v_reset (float): Reset voltage
        v_peak (float): Peak/threshold voltage
        i_bias (float): Neuron bias current
        t_r (float): Synaptic rise time
        t_d (float): Synaptic decay time
        dim (int): Dimension of target dynamics
        N (int): Number of neurons in network
        omega (np.ndarray): Reservoir connectivity matrix
        eta (np.ndarray): Feedback connectivity matrix (encoder)
        psi (np.ndarray): Input connectivity matrix
        phi (np.ndarray): Output connectivity matrix (decoder)
        refr_timer (): Timer for neurons in refractory period
        v (np.ndarray): Neuron voltage
        spikes (np.ndarray): Boolean array of neurons that spiked in last time step
        i_ps (np.ndarray): Postsynaptic current
        h (np.ndarray): Current filter
        i (np.ndarray): Neuronal current
        h_rate (np.ndarray): Spike filter
        R (np.ndarray): Filtered spike current
        x_hat (np.ndarray or float): Network output
    '''

    def __init__(self,
            N: int,
            omega: np.ndarray,
            eta: np.ndarray,
            t_m: float,
            t_ref: float,
            v_reset: float,
            v_peak: float,
            i_bias: float,
            t_r: float,
            t_d: float,
            dim: int=1,
            psi: np.ndarray=None
        ):
        '''
        '''

        #### LIF Variables
        self.t_m = t_m # Membrane time constant
        self.t_ref = t_ref # Refractory period
        self.v_reset = v_reset # Reset voltage
        self.v_peak = v_peak # Peak/threshold voltage
        self.i_bias = i_bias # Bias current
        self.t_r = t_r # Synaptic rise time
        self.t_d = t_d # Synaptic decay time
        self.dim = dim # Dimension of supervisor

        #### Connectivity Variables
        self.N = N # Network size
        self.omega = omega # Resevoir connections
        self.eta = eta # Feedback connections
        self.psi = psi # Input connections
        # Decoder
        if dim == 1:
            self.phi = np.zeros(N)
        else:
            self.phi = np.zeros((self.dim, self.N))

        #### State Variables
        self.refr_timer = np.zeros(N) # Timer for spike refractor periods
        self.v = self.v_reset + np.random.rand(N) * (30 - self.v_reset) # Neuron voltage
        self.spikes = self.v >= self.v_peak
        self.i_ps = np.zeros(N) # Postsynaptic current
        self.h = np.zeros(N) # Current filter
        self.i = np.zeros(N) # Neuronal current
        self.h_rate = np.zeros(N) # Spike filter
        self.R = np.zeros(N) # Output currents/neural basis
        self.x_hat = np.matmul(self.phi, self.R) # Readout

    def step(self, dt: float, i_in: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        '''Takes a step of size dt for the recursive network.

        Use the forward Euler intergration step to compute the state of the LIF
        network after a time step of dt.

        Args:
            dt: Size of integration time step
            i_in: Input currents to network

        Returns:
            x_hat: Readout of network
            R: Activation of network output
        '''

        # Get external input current if task has input
        if self.psi is not None and i_in is not None:
            i_in_ps = i_in * self.psi if len(self.psi.shape) == 1 else np.matmul(self.psi, i_in)
        else:
            i_in_ps = 0
        # Compute input current to each neuron
        if self.dim == 1:
            self.i = self.i_ps + self.eta * self.x_hat + self.i_bias + i_in_ps
        else:
            self.i = self.i_ps + np.matmul(self.eta, self.x_hat) + self.i_bias + i_in_ps

        # Update refractory timers
        self.refr_timer = np.maximum(0, self.refr_timer-dt)
        # Compute neuronal voltages
        self.v += dt * ((self.refr_timer<=0) * (self.i - self.v) / self.t_m)
        # Get spiking neurons
        self.spikes = self.v >= self.v_peak
        spike_idx = np.argwhere(self.spikes).flatten()
        spike_current = np.sum(self.omega[:, self.spikes], axis=1)
        # Set refractory timer
        self.refr_timer[spike_idx] = self.t_ref

        # Compute exponential filter
        if self.t_r == 0: # Single exponential filter
            self.i_ps = self.i_ps * np.exp(-dt/self.t_d) + spike_current * np.any(self.spikes) / self.t_d
            self.R = self.R * np.exp(-dt/self.t_d) + self.spikes / self.t_d
        else: # Double exponential filter
            self.i_ps = self.i_ps * np.exp(-dt/self.t_r) + dt * self.h
            self.h = self.h * np.exp(-dt/self.t_d) + spike_current * np.any(self.spikes) / (self.t_d * self.t_r)
            self.R = self.R * np.exp(-dt/self.t_r) + dt * self.h_rate
            self.h_rate = self.h_rate * np.exp(-dt/self.t_d) + self.spikes / (self.t_r * self.t_d)

        # Interpolant implementation
        self.v[spike_idx] = self.v_reset + self.v[spike_idx] - self.v_peak
        # Compute output
        self.x_hat = np.matmul(self.phi, self.R)

        return self.x_hat, self.R

    def simulate(self, dt: float, N: int, I_in: np.ndarray=None, save_values: dict={}) -> Tuple[np.ndarray, dict]:
        '''Simulates a series of steps for the spike based LIF network.

        Uses the forward Euler integration step to simulate the LIF-spike network
        for a given number of steps. The attributes of the network named in save_values
        are saved at each time step and returned as numpy ndarrays in a dictionary.
        The dict save_values should consist of keys (str) indicating the name of the
        attribute to save and a value (int/None) to indicate how many neuron samples to save.
        If the given value is an int then that number of neuron samples will be saved
        and if it is None then parameter for all neurons will be saved.

        Args:
            dt: Size of integration time step
            N: Number of integration steps to take
            i_in: Input currents to network
            save_values: Name of network attributes and the number of neurons to save for

        Returns:
            x_hat_save: Timeseries of network output
            saves: Dictionary of timeseries for desired values

        Examples:
            >>> net = LIF_Rate(N=4, **params) # A network of 4 neurons
            >>> net.simulate(0.1, 4)
            np.ndarray([0,0,0,0]), {}
            >>> net.simulate(0.1, 4, save_values={'R':None})
            np.ndarray([0,0,0,0]), {'R':np.ndarray([[1,2,3,4],[5,6,7,8],[2,4,5,6],[1,2,3,4]])}
            >>> net.simulate(0.1, 4, save_values={'R':2})
            np.ndarray([0,0,0,0]), {'R':np.ndarray([[1,2,3,4],[5,6,7,8]])}
        '''

        if not all(k in vars(self) for k in save_values.keys()):
            raise ValueError('Given save value is not an attribute of network.')

        # Save structures
        x_hat_save = np.zeros(N) if self.dim==1 else np.zeros((N, self.dim))
        saves = {
            value : np.zeros((N))
                    if num_save is None else
                    np.zeros((N, num_save))
                    for value, num_save in save_values.items()
        }

        # Precompute integration factors
        e_t_r = np.exp(-dt/self.t_r) if self.t_r != 0 else 0
        e_t_d = np.exp(-dt/self.t_d)

        for n in range(N):
            # Get external input current if task has input
            if self.psi is not None and I_in is not None:
                i_in_ps = I_in[n] * self.psi if len(self.psi.shape) == 1 else np.matmul(self.psi, I_in[n])
            else:
                i_in_ps = 0

            # Compute input current to each neuron
            if self.dim == 1:
                self.i = self.i_ps + self.eta * self.x_hat + self.i_bias + i_in_ps
            else:
                self.i = self.i_ps + np.matmul(self.eta, self.x_hat) + self.i_bias + i_in_ps

            # Compute neuronal voltages with refractory period
            self.refr_timer = np.maximum(0, self.refr_timer-dt)
            self.v += dt * ((self.refr_timer<=0) * (self.i - self.v) / self.t_m)
            # Get spiking neurons
            self.spikes = self.v >= self.v_peak
            spike_idx = np.argwhere(self.spikes).flatten()
            spike_current = np.sum(self.omega[:, self.spikes], axis=1)
            # Set refractory timer
            self.refr_timer[spike_idx] = self.t_ref

            # Compute exponential filter
            if self.t_r == 0: # Single exponential filter
                self.i_ps = self.i_ps * e_t_d + spike_current * np.any(self.spikes) / self.t_d
                self.R = self.R * e_t_d + self.spikes / self.t_d
            else: # Double exponential filter
                self.i_ps = self.i_ps * e_t_r + dt * self.h
                self.h = self.h * e_t_d + spike_current * np.any(self.spikes) / (self.t_d * self.t_r)
                self.R = self.R * e_t_r + dt * self.h_rate
                self.h_rate = self.h_rate * e_t_d + self.spikes / (self.t_r * self.t_d)

            # Interpolant implementation
            self.v[spike_idx] = self.v_reset + self.v[spike_idx] - self.v_peak

            # Save outputs
            self.x_hat = np.matmul(self.phi, self.R)
            x_hat_save[n] = self.x_hat
            for value, num_save in save_values.items():
                if num_save is None:
                    saves[value][n] = getattr(self, value)
                else:
                    saves[value][n] = getattr(self, value)[:num_save]

        return x_hat_save, saves

    def get_state(self) -> dict:
        state = {
            'v' : self.v,
            'refr_timer' : self.refr_timer,
            'i_ps' : self.i_ps,
            'h' : self.h,
            'h_rate' : self.h_rate,
            'R' : self.R,
            'x_hat' : self.x_hat
        }

        return state

    def set_state(self, new_state: dict) -> None:
        self.v = new_state['v']
        self.refr_timer = new_state['refr_timer']
        self.i_ps = new_state['i_ps']
        self.h = new_state['h']
        self.h_rate = new_state['h_rate']
        self.R = new_state['R']
        self.x_hat = new_state['x_hat']

    def get_decoder(self) -> np.ndarray:
        return self.phi.copy()

    def set_decoder(self, new_phi: np.ndarray) -> None:
        self.phi = new_phi

class LIF_Rate(Abstract_Network):
    '''Simulates firing rate based leaky-integrate-and-fire reservoir neural network.

    Attributes:
        t_m (float): Membrane time constant
        t_ref (float): Refractor period length
        v_reset (float): Reset voltage
        v_peak (float): Peak/threshold voltage
        i_bias (float): Neuron bias current
        t_r (float): Synaptic rise time
        t_d (float): Synaptic decay time
        dim (int): Dimension of target dynamics
        N (int): Number of neurons in network
        omega (np.ndarray): Reservoir connectivity matrix
        eta (np.ndarray): Feedback connectivity matrix (encoder)
        psi (np.ndarray): Input connectivity matrix
        phi (np.ndarray): Output connectivity matrix (decoder)
        i_ps (np.ndarray): Postsynaptic current
        h (np.ndarray): Current filter
        i (np.ndarray): Neuronal current
        h_rate (np.ndarray): Spike filter
        R (np.ndarray): Filtered spike current
        x_hat (np.ndarray or float): Network output
    '''

    def __init__(self,
            N: int,
            omega: np.ndarray,
            eta: np.ndarray,
            t_m: float,
            t_ref: float,
            v_reset: float,
            v_peak: float,
            i_bias: float,
            t_r: float,
            t_d: float,
            dim: int=1,
            psi: np.ndarray=None
        ):
        #### LIF Variables
        self.t_m = t_m # Membrane time constant
        self.t_ref = t_ref # Refractory period
        self.v_reset = v_reset # Reset voltage
        self.v_peak = v_peak # Peak/threshold voltage
        self.i_bias = i_bias # Bias current
        self.t_r = t_r # Synaptic rise time
        self.t_d = t_d # Synaptic decay time
        self.dim = dim # Dimension of supervisor

        #### Connectivity Variables
        self.N = N # Network size
        self.omega = omega # Resevoir connections
        self.eta = eta # Feedback connections
        self.psi = psi # Input connections
        # Decoder
        if dim == 1:
            self.phi = np.zeros(N)
        else:
            self.phi = np.zeros((self.dim, self.N))

        #### State Variables
        self.i_ps = self.i_bias * (np.random.rand(N) - 0.5) # Postsynaptic current
        self.h = np.zeros(N) # Synaptic filter
        self.h_rate = np.zeros(N) #  Synaptic filter
        self.R = np.zeros(N) # Output current/neural basis
        self.x_hat = np.matmul(self.phi, self.R) # Readout

    def step(self, dt: float, i_in: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        '''Takes a step of size dt for the recursive network.

        Use the forward Euler intergration step to compute the state of the rate
        based LIFnetwork after a time step of dt.

        Args:
            dt: Size of integration time step
            i_in: Input currents to network

        Returns:
            x_hat: Output of network
            R: Activation of network output
        '''

        # Get external input current if task has input
        if self.psi is not None and i_in is not None:
            i_in_ps = i_in * self.psi if len(self.psi.shape) == 1 else np.matmul(self.psi, i_in)
        else:
            i_in_ps = 0
        # Compute input current to each neuron
        if self.dim == 1:
            self.i = self.i_ps + self.eta * self.x_hat + self.i_bias + i_in_ps
        else:
            self.i = self.i_ps + np.matmul(self.eta, self.x_hat) + self.i_bias + i_in_ps

        # Compute firing rate equivalent output current for each neuron
        self.rate = np.nan_to_num(1/(self.t_r - self.t_m * (np.log(self.i - self.v_peak) - np.log(self.i - self.v_reset))), copy=False)
        spike_current = dt * np.matmul(self.omega, self.rate)

        # Compute exponential filter
        if self.t_r == 0: # Single exponential filter
            self.i_ps = self.i_ps * np.exp(-dt/self.t_d) + spike_current / self.t_d
            self.R = self.R * np.exp(-dt/self.t_d) + dt * self.rate / self.t_d
        else: # Double exponential filter
            self.i_ps = self.i_ps * np.exp(-dt/self.t_r) + dt * self.h
            self.h = self.h * np.exp(-dt/self.t_d) + spike_current / (self.t_d * self.t_r)
            self.R = self.R * np.exp(-dt/self.t_r) + dt * self.h_rate
            self.h_rate = self.h_rate * np.exp(-dt/self.t_d) + dt * self.rate / (self.t_r * self.t_d)

        # Compute output
        self.x_hat = np.matmul(self.phi, self.R)

        return self.x_hat, self.R

    def simulate(self, dt: float, N: int, I_in: np.ndarray=None, save_values: dict={}) -> Tuple[np.ndarray, dict]:
        '''Simulates a series of steps for the rate based LIF network.

        Uses the forward Euler integration step to simulate the LIF-rate network
        for a given number of steps. The attributes of the network named in save_values
        are saved at each time step and returned as numpy ndarrays in a dictionary.
        The dict save_values should consist of keys (str) indicating the name of the
        attribute to save and a value (int/None) to indicate how many neuron samples to save.
        If the given value is an int then that number of neuron samples will be saved
        and if it is None then parameter for all neurons will be saved.

        Args:
            dt: Size of integration time step
            N: Number of integration steps to take
            i_in: Input currents to network
            save_values: Name of network attributes and the number of neurons to save for

        Returns:
            x_hat_save: Timeseries of network output
            saves: Dictionary of timeseries for desired values

        Examples:
            >>> net = LIF_Rate(N=4, **params) # A network of 4 neurons
            >>> net.simulate(0.1, 4)
            np.ndarray([0,0,0,0]), {}
            >>> net.simulate(0.1, 4, save_values={'R':None})
            np.ndarray([0,0,0,0]), {'R':np.ndarray([[1,2,3,4],[5,6,7,8],[2,4,5,6],[1,2,3,4]])}
            >>> net.simulate(0.1, 4, save_values={'R':2})
            np.ndarray([0,0,0,0]), {'R':np.ndarray([[1,2,3,4],[5,6,7,8]])}
        '''

        if not all(k in vars(self) for k in save_values.keys()):
            raise ValueError('Given save value is not an attribute of network.')

        # Save structures
        x_hat_save = np.zeros(N) if self.dim==1 else np.zeros((N, self.dim))
        saves = {
            value : np.zeros((N))
                    if num_save is None else
                    np.zeros((N, num_save))
                    for value, num_save in save_values.items()
        }

        # Precompute integration factors
        e_t_r = np.exp(-dt/self.t_r) if self.t_r != 0 else 0
        e_t_d = np.exp(-dt/self.t_d)

        for n in range(N):
            # Get external input current if task has input
            if self.psi is not None and I_in is not None:
                i_in_ps = I_in[n] * self.psi if len(self.psi.shape) == 1 else np.matmul(self.psi, I_in[n])
            else:
                i_in_ps = 0
            # Compute input current to each neuron
            if self.dim == 1:
                self.i = self.i_ps + self.eta * self.x_hat + self.i_bias + i_in_ps
            else:
                self.i = self.i_ps + np.matmul(self.eta, self.x_hat) + self.i_bias + i_in_ps

            # Compute firing rate equivalent output current for each neuron
            self.rate = np.nan_to_num(1/(self.t_r - self.t_m * (np.log(self.i - self.v_peak) - np.log(self.i - self.v_reset))), copy=False)
            self.spike_current = dt * np.matmul(self.omega, self.rate)

            # Compute exponential filter
            if self.t_r == 0: # Single exponential filter
                self.i_ps = self.i_ps * e_t_d + self.spike_current / self.t_d
                self.R = self.R * e_t_d + dt * self.rate / self.t_d
            else: # Double exponential filter
                self.i_ps = self.i_ps * e_t_r + dt * self.h
                self.h = self.h * e_t_d + self.spike_current / (self.t_d * self.t_r)
                self.R = self.R * e_t_r + dt * self.h_rate
                self.h_rate = self.h_rate * e_t_d + dt * self.rate / (self.t_r * self.t_d)

            # Save outputs
            self.x_hat = np.matmul(self.phi, self.R)
            x_hat_save[n] = self.x_hat
            for value, num_save in save_values.items():
                if num_save is None:
                    saves[value][n] = getattr(self, value)
                else:
                    saves[value][n] = getattr(self, value)[:num_save]

        return x_hat_save, saves

    def get_state(self) -> dict:
        state = {
            'i_ps' : self.i_ps,
            'h' : self.h,
            'h_rate' : self.h_rate,
            'R' : self.R,
            'x_hat' : self.x_hat
        }

        return state

    def set_state(self, new_state: dict) -> None:
        self.i_ps = new_state['i_ps']
        self.h = new_state['h']
        self.h_rate = new_state['h_rate']
        self.R = new_state['R']
        self.x_hat = new_state['x_hat']

    def get_decoder(self) -> np.ndarray:
        return self.phi.copy()

    def set_decoder(self, new_phi: np.ndarray) -> None:
        self.phi = new_phi

