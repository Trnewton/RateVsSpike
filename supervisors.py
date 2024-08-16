from typing import Iterable, Tuple
import numpy as np


def song_from_code(song_code: Iterable, note_form: np.ndarray, num_pitch: int, HDTS_flag: bool=False) -> np.ndarray:
    '''Generates a song timeseries from a list of notes.

    Takes a series of notes represented by list of lists of integers. In each
    sub-list, the integer represents which pitch the notes should be which is
    then encoded by a different row of the time series matrix.

    Args:
        song_code: List of notes in the song
        note_form: Array of waveform for notes pulses
        num_pitch: Number of different pitches used in song code
        HDTS_flag: Flag to indicate using HDTS or not

    Returns:
        song: Resulting time series of song

    Examples:
        >>> print(song_from_code([[0],[1,2]], np.array([3,4]), 3))
        [[3,4,0,0],
        [0,0,3,4],
        [0,0,3,4]]

        >>> print(song_from_code([[0],[1,2],[0]], np.array([-4,-5,-4]), 4, True))
        [[-4,-5,-4,0,0,0,-4,-5,-4],
        [0,0,0,-4,-5,-4,0,0,0],
        [0,0,0,-4,-5,-4,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [-4,-5,-4,0,0,0,0,0,0],
        [0,0,0,-4,-5,-4,0,0,0],
        [0,0,0,0,0,0,-4,-5,-4]]
    '''

    song = (np.zeros((num_pitch + len(song_code), len(song_code)*len(note_form)))
            if HDTS_flag
            else np.zeros((num_pitch, len(song_code)*len(note_form))))

    for n, notes in enumerate(song_code):
        if HDTS_flag:
            song[num_pitch + n, n*len(note_form):(n+1)*len(note_form)] = note_form
        for note in notes:
            song[note, n*len(note_form):(n+1)*len(note_form)] = note_form

    return song

def ode_short(dt:float, HDTS_flag:bool=False) -> np.ndarray:
    '''Generates the Ode to Joy supervisor.'''

    # First bar of Ode to Joy notes
    ODE_CODE = [[2],[2],[1],[0],[0],[1],[2],[3],[4],[4],[3],[2],[2],[3],[3],[2]]
    ODE_PITCH = 5

    # Note waveforms
    fq = 2
    T_q = np.arange(0, 1/(2*fq), dt)
    quater = np.sin(2*np.pi * fq * T_q)
    T_h = np.arange(0, 1/fq, dt)
    half = np.sin(fq*np.pi * T_h)
    # Create song supervisor
    song = song_from_code(ODE_CODE[:16], quater, ODE_PITCH, HDTS_flag)
    # Add half note
    song[:ODE_PITCH,-half.shape[0]:] = 0
    song[3,-half.shape[0]:] = half
    sup = song.T

    return sup

def sin_sup(dt: float, fq: float) -> np.ndarray:
    '''Generators a sinusodial supervisor.'''

    T = np.arange(0, 1/fq, dt)
    sup_arr = np.sin(2 * np.pi * fq * T)

    return sup_arr

def fourier_sup(f_0, n, dt) -> np.ndarray:
    '''Generates the Fourier supervisors'''

    F = f_0 * np.arange(1, n)
    T = np.arange(0, 1/np.min(F), dt)
    sup = np.sin(2 * np.pi * np.outer(T, F))

    return sup

def pitchfork(
        dt: float,
        gamma: float,
        min_pulse_lenght: int,
        max_pulse_lenght: int,
        min_start_time: int,
        max_start_time: int,
        mean_pulse_height: int,
        num_pulse: int,
        seed: int=None
    ) -> Tuple[np.ndarray,np.ndarray]:
    '''Generates a time series of perturbations and resulting Pitchfork dynamics.

    Args:
        dt: Integration time step size
        gamma: Pitchfork time constant
        min_pulse_lenght: Minimum length of perturbation in integration steps
        max_pulse_lenght: Maximum length of perturbation in integration steps
        min_start_time: Minimum time between perturbations in integration steps
        max_start_time: Maximum time between perturbations in integration steps
        mean_pulse_height: Average height of perturbations
        num_pulse: Number of perturbations
        seed: Random seed used

    Returns:
        sup: Timeseries of resulting pitchfork dynamics
        pert:Timeseries of perturbations
    '''

    if seed is not None:
        np.random.seed(seed)

    # Lengths of each perturbation
    lengths = np.random.randint(low=min_pulse_lenght, high=max_pulse_lenght, size=num_pulse)
    # start time of perturbation after last
    start_times = np.random.randint(low=min_start_time, high=max_start_time, size=num_pulse)
    # strength of perterbation
    pulse_heights = mean_pulse_height * np.random.randn(num_pulse)

    start_idx = start_times + np.cumsum(start_times, axis=0)
    end_idx = start_idx + lengths

    pert = np.zeros(np.max(end_idx[-1]) + 1)

    for n in range(num_pulse):
        pert[start_idx[n]:end_idx[n]] = pulse_heights[n]

    sup = np.zeros(pert.shape[0])

    for n, x in enumerate(pert):
        sup[n] = sup[n-1] + ((1 - sup[n-1] * sup[n-1]) * sup[n-1] + x) * dt / gamma

    return sup, pert

def pitchfork_sup(dt: float) -> Tuple[np.ndarray,np.ndarray]:
    '''Generates pitchfork supervisor with given intergration step size.'''

    # Pitchfork time constant
    gamma = 0.01

    # Perterbation parameters
    # Min and max length a perturbation lasts
    min_pulse_lenght = int(0.005/dt)
    max_pulse_lenght = int(0.01/dt)
    # Min and max time between perturbations
    min_start_time = int(0.1/dt)
    max_start_time = int(0.5/dt)
    # Mean height of perturbation
    mean_pulse_height = 2
    # Number of perturbations
    num_pulse = 400

    sup, pert = pitchfork(
        dt, gamma, min_pulse_lenght, max_pulse_lenght, min_start_time,
        max_start_time, mean_pulse_height, num_pulse, seed=2
    )

    return sup, pert

def get_supervisor(sup_str: str, dt: float) -> Tuple[np.ndarray,np.ndarray]:
    '''Generates the desired supervisor dynamics and input current for given time step.

    Generates one of sin, Ode to Joy, Ode to Joy with HDTS, Fourier, or pitchfork
    supervisors. The supervisor options and corresponding sup_str strings are:
        'sin_[n]'
            Sin wave with frequency [n]
        'ode'
            First bar of Ode to Joy
        'ode_HDTS'
            First bar of Ode to Joy with HDTS
        'fourier_[n]'
            10 dimensional supervisor of sin waves with frequencies {1*[n], 2*[n],...,10*[n]}
        'pitchfork'
            Pitchfork dynamical system with random perturbations

    Args:
        sup_str: Name of the supervisor that is to be generated
        dt: Integration time constant

    Returns:
        sup_arr: Supervisor timeseries
        I_in: Input current timeseries or None
    '''

    sup_list = [word.strip() for word in sup_str.split('_')]
    I_in = None

    if sup_list[0] == 'sin':
        fq = float(sup_list[1])
        sup_arr = sin_sup(dt, fq)
    elif sup_str == 'ode':
        sup_arr = ode_short(dt)
    elif sup_str == 'ode_HDTS':
        sup_arr = ode_short(dt, HDTS_flag=True)
    elif sup_list[0] == 'fourier':
        sup_arr = fourier_sup(float(sup_list[1]), 10, dt)
    elif sup_str == 'pitchfork':
        sup_arr, I_in = pitchfork_sup(dt)

    return sup_arr, I_in