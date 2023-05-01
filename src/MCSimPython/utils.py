# Utility functions

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2022-10-12
# Revised: 2023-02-09 Harald Mo    Add from/to 6 and 3DOF functions
# 
# Tested:  See tests/test_utils.py
# 
# Copyright (C) 202x: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
from time import time
from scipy.signal import welch
import matplotlib.pyplot as plt
import re
import os
import json

dof3_matrix_mask = np.ix_([0, 1, 5], [0, 1 ,5])
dof3_array = np.ix_([0, 1, 5])

def complex_to_polar(complex_values):
    """Complex value to polar coordinates.
    
    Parameters
    ----------
    complex_values : flot, array_like
        Complex value to be transformed.

    Returns
    -------
    amp : float
        Amplitude/radius
    theta : float
        Angle in rad.
    """
    complex_values = np.asarray_chkfinite(complex_values)
    amp = np.abs(complex_values)
    theta = np.angle(complex_values)
    return amp, theta


def polar_to_complex(amp, theta):
    """Polar coordinate to complex value.
    
    The complex value is calculated from the polar
    coordinates as 

    .. math::
       z = A (\\cos(\\theta) + j\\sin(\\theta))
    
    Parameters
    ----------
    amp : float
        Amplitude/radius of polar value.
    theta : float
        Angle in radians.

    Returns
    -------
    complex_value : complex
        Complex value.
    
    Examples
    --------

    >>> from MCSimPython.utils import polar_to_complex
    >>> amp, theta = 1.0, 0.0
    >>> polar_to_complex(amp, theta)
    (1 + 0j)
    
    >>> amp, theta = 1.0, np.pi
    >>> polar_to_complex(amp, theta)
    (-1 + 0j)

    """
    amp = np.asarray_chkfinite(amp)
    theta = np.asarray_chkfinite(theta)
    return amp*(np.cos(theta) + 1j * np.sin(theta))


def pipi(theta):
    """Return angle in range [-pi, pi).
    
    Parameters
    ----------
    theta : float, array_like
        Angle in radians to be mapped to [-pi, pi).

    Returns
    -------
    angle : float, array_like
        Smallest signed angle in range [-pi, pi).

    Examples
    --------

    >>> from MCSimPython.utils import pipi
    >>> import numpy as np
    >>> angle = 2*np.pi
    >>> pipi(angle)
    0.0

    >>> angle = np.deg2rad(270)
    >>> pipi(angle)
    -1.570796

    """
    return np.mod(theta + np.pi, 2*np.pi) - np.pi


def to_positive_angle(theta):
    """Map angle from [-pi, pi) to [0, 2*pi).
    
    Parameters
    ----------
    theta : array_like
        Angle in radians in [-pi, pi).
    
    Returns
    -------
    out : array_like
        Angle in [0, 2*pi).

    Note
    ----
    The function does not calculate the smallest
    signed positive angle if the input is outside
    [-pi, pi).

    See Also
    --------
    MCSimPython.utils.pipi : 
        Map angle to [-pi, pi).

    Examples
    --------

    >>> from MCSimPython.utils import to_positive_angle
    >>> import numpy as np
    >>> angle = -np.pi
    >>> to_positive_angle(angle)
    array(3.14159265)
    
    Example of wrong use
    
    >>> angle = 2.5*np.pi # 7.85398163
    >>> to_positive_angle(angle)
    array(7.85398163)

    Correct use for angle > 2pi

    >>> from MCSimPython.utils import pipi
    >>> angle = 2.5*np.pi
    >>> to_positive_angle(pipi(angle))
    array(1.57079633)

    """
    return np.where(theta < 0, theta + 2*np.pi, theta)


def pipi2cont(psi, psi_prev):
    """Lifting algorithm."""
    arr = np.array([psi_prev, psi])
    return np.unwrap(arr)[-1]


def Rx(phi):
    """3DOF Rotation matrix about x-axis."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])


def Ry(theta):
    """3DOF Rotation matrix about y-axis."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def Rz(psi):
    """3DOF Rotation matrix about z-axis."""
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])


def Rzyx(eta):
    """Full roation matrix."""
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]
    return Rz(psi)@Ry(theta)@Rx(phi)


def Tzyx(eta):
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]
    return np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])


def J(eta):
    """6 DOF rotation matrix."""
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    return np.block([
        [Rzyx(eta), np.zeros((3, 3))],
        [np.zeros((3, 3)), Tzyx(eta)]
    ])


def Smat(x):
    """
    Skew-symmetric cross-product operator matrix.

    Parameters
    ----------
    x: 3x1-array

    Return
    ------
    S: 3x3-array
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def three2sixDOF(v):
    """3 to 6 DOF conversion.
    
    Converts a vector a matrix from 3DOF
    to 6DOF.

    Parameters
    ----------
    v : array_like
        Vector or matrix to be converted.

    Returns
    -------
    out : array_like
        Vector (6,1) or matrix (6, 6) array.
    """
    if v.ndim == 2:   # Matrix
        v = np.concatenate(v, axis=None)
        v = np.concatenate((v[0:2], np.zeros(3), v[2:5], np.zeros(3), v[5], np.zeros(18), v[6:8], np.zeros(3), v[8]), axis=None).reshape((6,6))
    elif v.ndim == 1: # Vector
        v = np.array([v[0], v[1], 0, 0, 0, v[2]])
    return v

def six2threeDOF(v):
    """6 to 3 DOF conversion.
    
    Convert a 6DOF vecor or matrix to 3DOF.

    Parameters
    ----------
    v : array_like
        6 DOF vector or matrix to be converted to 3DOF.
    
    Returns
    -------
    out : array_like
        Vector (3,1) or matrix (3,3) array.
    """
    if v.ndim == 2:   # Matrix
        i = np.ix_([0,1,5],[0,1,5])
        v = v[i]  
    elif v.ndim == 1: # Vector
        i = np.ix_([0,1,5])  
        v = v[i]
    return v



def timeit(func):
    """
    Decorator for measuring execution time of function.

    Print the execution time of a function. (Mainly for
    debugging and analyzis purposes).

    Parameters
    ----------
    func : function
        Function to be timed.

    
    Examples
    --------
    >>> from MCSimPython.utils import timeit
    >>> @timeit
    ... def loop(n):
    ...     for i in range(n):
    ...         print(i)
    ...
    >>> loop(5)
    0
    1
    2
    3
    4
    Execution time of loop: 0.004         

    """
    def wrapper(*args, **kwargs):
        t1 = time()
        results = func(*args, **kwargs)
        t2 = time()
        print(f"Execution time of {func.__name__}: {(t2 - t1):.4f}")
        return results
    return wrapper


def rigid_body_transform(r, eta, in_ned=True):
    """Calculate the relative motion of a point r different from the COG.

    The calculation assumes small angles (s.t. cos(theta)=0 and sin(theta)=theta)
    and is computed as:

    ``s = (\eta_1, \eta_2, \eta_3)^T + (\eta_4, \eta_5, \eta_6) x (r_x, r_y, r_z)``
    
    Parameters
    ----------
    r : array_like
        Lever arm from COG to point of interest
    eta : array_like
        6DOF vessel pose (surge, sway, heave, roll, pitch, yaw)
    in_ned : bool (default = False)
        Reference frame definition of eta. If True, the vessel
        pose eta is assumed to be defined in the NED-frame. If False
        eta is assumed to be defined in body-frame.

    Returns
    -------
    s : array_like
        Translation vector which is the same as (delta_x, delta_y, delta_z).
    """

    if in_ned:
        eta = np.copy(np.linalg.inv(J(eta))@eta)
        print(eta)
    return eta[:3] + np.cross(eta[3:], r)


def power_spectral_density(timeseries, fs, freq_hz=False, nperseg=2**11):
    """Compute the Power Spectral Density (PSD) of a timeseries.
    
    The PSD is calculated using scipy.signals.welch

    Parameters
    ----------
    timerseries : array_like
        Timeseries array.
    fs : int
        Sampling frequency
    freq_hz : bool (optional)
        Calculate the PSD in [hz]. Defaults to False.
    nperseg : float (optional)
        Number of points per segments used in the scipy.signal.welch function.
    
    Returns
    -------
    f : array_like
        Frequencies in PSD. Returned as [rad/s] if freq_hz=False.
    PSD : array_like
        PSD of timeseries. Returned as [unit/(rad/s)] if freq_hz=False
    """
    f, S_f = welch(timeseries, fs=fs, nperseg=nperseg)
    if not freq_hz:
        return f*2*np.pi, S_f/(2*np.pi)
    return f, S_f


# --------- READ DATA UTILITY FUNCTIONS -------------------
re_science = "\s{1,}-?\d\.?\d*[Ee][+\-]?\d+"
re_int = "\s{1,}[0-9]{1,}"
re_float = "\s{1,}[0-9]{1,}\.[0-9]{1,}"

def data2num(line):
    return (float(x) for x in re.findall(re_science, line))


def data2int(line):
    return (int(x) for x in re.findall(re_int, line))


def data2float(line):
    return (float(x) for x in re.findall(re_float, line))


def read_tf(file_path, tf_type="motion"):
    """Read VERES transfer function output.
    
    The function reads data from veres input files. 
    - Motion RAOs are found in '.re1'
    - Force RAOs are found in '.re8'

    The RAOs are converted from the global coordinate system in VERES, which
    is defined as: x-axis positive to stern, y-axis postive to stbd, and 
    z-axis positive up, to the body frame used on `MCSimPython`.
    
    Parameters
    ----------
    file_path : string
        The path to the .re1 or .re8 file.
    tf_type : string (default='motion')
        Type of transfer function. Can be either force or motion RAO.
        If type = 'motion', the input file should be .re1, else
        the expected file is .re8 for force RAO.

    Returns
    -------
    rho_w : float
        Water density
    g : float
        Gravitational acceleration
    freqs : array_like
        Array of n frequencies in [rad/s] for which the RAOs are calculated.
    headings : array_like
        Array of m headings in [rad] for which the RAOs are calculated.
    velocities : array_like
        Array of j velocities [m/s] for which the RAOs are calculated.
    rao_complex : array_like
        Array of complex RAOs with dimension (6, n, m, j). Using dtype=np.complex128.
    rao_amp : array_like
        Array of RAO amplitudes [m/m] with dimension (6, n, m, j).
    rao_phase : array_like
        Array of RAO phase [rad] with dimension (6, n, m, j).

    Note
    ----
    The wave direction convention of VERES is different from `MCSimPython`.
    VERES use 0 deg as head sea, and 180 deg as following sea. While `MCSimPython` use
    head sea = 180 deg, and following sea = 0 deg. 
    """
    
    fid = open(file_path, 'r')
    header = []
    for i in range(6):
        header.append(fid.readline())

    data_info = []
    for i in range(3):
        data_info.append(fid.readline())

    rho_w, g = data2num(data_info[0])
    Lpp, breadth, draught = data2num(data_info[1])
    LCG, VCG = data2num(data_info[2])

    print("Vessel Parameters".center(100, '-'))
    print(f"Water Density: {rho_w} [kg/m^3].")
    print(f"Acceleration of gravity: {g} [m/s^2]")
    print(f"Length between the perpendiculars: {Lpp} [m]")
    print(f"Breadth: {breadth} [m]")
    print(f"Draught: {draught} [m]")
    print(f"Vertical center of gravity (rel. BL): {VCG} [m]")
    print(f"Longitudinal center of gravity (rel. Lpp/2): {LCG} [m]")
    print("".center(100, '-'))


    run_info = fid.readline()
    novel, nohead, nofreq, nodof = data2int(run_info)

    freqs = np.zeros(nofreq)
    headings = np.zeros(nohead)
    velocities = np.zeros(novel)
    rao_complex = np.zeros((nodof, nofreq, nohead, novel), dtype=np.complex128)
    rao_amp = np.zeros((nodof, nofreq, nohead, novel), dtype=np.float64)
    rao_phase = np.zeros_like(rao_amp, dtype=np.float64)

    for v in range(novel):
        k = 0
        velocity, *_ = data2float(fid.readline())
        velocities[v] = velocity
        for h in range(nohead):
            temp_h = fid.readline()
            heading, *_ = data2float(temp_h)
            headings[h] = np.deg2rad(heading)
            
            for f in range(nofreq):
                freq_f, *_ = data2float(fid.readline())
                freqs[f] = freq_f
                k = k+1
                for dof in range(6):
                    temp = fid.readline()
                    d, *_ = data2int(temp)
                    real, img, *_ = data2num(temp)
                    rao_complex[dof, f, h, v] = real + 1j*img
                    rao_amp[dof, f, h, v] = np.sqrt(real**2 + img**2)
                    rao_phase[dof, f, h, v] = np.arctan2(img, real)

    fid.close()
    # 1) Convert the RAOs from VERES frame to body-frame.
    # 2) Make sure that we have raos for the full range 0-360 deg.

    # Create a transformation vector. 
    T = J([0., 0., 0., 0., np.pi, 0.])@np.ones(6)
    T = np.array([-1, 1, -1, -1, 1, -1])

    # Flip headings such that they correspond with 
    # relative wave heading convention of MCSimPython
    # Note, we need to do this for heading 0 -> heading 180 deg.
    rao_complex_n = np.flip(rao_complex, axis=2)
    rao_amp_n = np.flip(rao_amp, axis=2)
    rao_phase_n = np.flip(rao_phase, axis=2)
    # headings_n = np.flip(headings)

    for dof in range(6):
        rao_phase_n[dof] = rao_phase_n[dof]*T[dof]
        if T[dof] < 0:
            rao_complex_n[dof] = np.conjugate(rao_complex_n[dof])

    return freqs, headings, velocities, rao_complex_n, rao_amp_n, rao_phase_n


def read_hydrod(filepath):
    f = open(filepath, 'r')
    header = []
    run_info = []
    for i in range(6):
        header.append(f.readline())
    for i in range(4):
        run_info.append(f.readline())

    rhow, g = data2num(run_info[0])
    lpp, breadth, draught = data2num(run_info[1])
    LCG, VCG = data2num(run_info[2])
    version, *_ = data2int(run_info[3])
    print(f"Version = {version}")

    novel, nohead, nofreq, nodof = data2int(f.readline())

    A = np.zeros((nodof, nodof, nofreq, novel))
    A_added = np.zeros_like(A)
    B = np.zeros_like(A)
    B_added = np.zeros_like(B)
    C = np.zeros_like(A)
    C_added = np.zeros_like(A)
    Bv44_linear = np.zeros((nofreq, nohead, novel))
    Bv44_nonlin = np.zeros_like(Bv44_linear)
    Bv44_linearized = np.zeros_like(Bv44_linear)

    Mrb = np.zeros((nodof, nodof))
    for i in range(nodof):
        Mrb[i] = [m for m in data2num(f.readline())]

    for v in range(novel):
        vel, *_ = data2float(f.readline())
        for h in range(nohead):
            head, *_ = data2float(f.readline())
            for j in range(nofreq):
                freq, *_ = data2float(f.readline())
                for k in range(nodof):
                    temp = f.readline()
                    a_kj = np.array([a for a in data2num(temp)])
                    A[k, :, j, v] = a_kj
                if version==2:
                    for k in range(nodof):
                        temp = f.readline()
                        A_added[k, :, j, v] = np.array([aa for aa in data2num(temp)])
                for k in range(nodof):
                    temp = f.readline()
                    b_kj = np.array([b for b in data2num(temp)])
                    B[k, :, j, v] = b_kj
                if version==2:
                    for k in range(nodof):
                        temp = f.readline()
                        B_added[k, :, j, v] = np.array([bb for bb in data2num(temp)])
                for k in range(nodof):
                    temp = f.readline()
                    c_kj = np.array([c for c in data2num(temp)])
                    C[k, :, j, v] = c_kj
                if version==2:
                    for k in range(nodof):
                        temp = f.readline()
                        C_added[k, :, j, v] = np.array([cc for cc in data2num(temp)])
                bv_l, bv_nl, bv_ll, *_ = data2num(f.readline())
                Bv44_linear[j, h, v] = bv_l
                Bv44_nonlin[j, h, v] = bv_nl
                Bv44_linearized[j, h, v] = bv_ll
    

    f.close()
    return Mrb, A, B, C, Bv44_linear, Bv44_nonlin, Bv44_linearized


def read_wave_drift(filepath):
    f = open(filepath, 'r')

    header = []
    run_info = []
    for i in range(6):
        header.append(f.readline())
    for i in range(3):
        run_info.append(f.readline())

    rhow, g = data2num(run_info[0])
    lpp, breadth, D = data2num(run_info[1])
    novel, nohead, nofreq = data2int(run_info[2])

    drift_frc = np.zeros((6, nofreq, nohead, novel))

    print("WAVE DRIFT DATA".center(100, '-'))
    print("Rho_w".ljust(50, ' ') + f"{rhow}")
    print("Lpp".ljust(50, ' ') + f"{lpp}")
    print(f"Novel: {novel}")
    print(f"Nohead: {nohead}")
    print(f"Nofreq: {nofreq}")
    print(".".center(100, '-'))

    for v in range(novel):
        for h in range(nohead):
            vel, head_i = data2float(f.readline())
            for i in range(nofreq):
                temp = f.readline()
                freq_, *_ = data2float(temp)
                addr, swdr, yawr, *_ = data2num(temp)
                drift_frc[0, i, h, v] = addr*rhow*g*breadth**2/lpp
                drift_frc[1, i, h, v] = swdr*rhow*g*breadth**2/lpp
                drift_frc[2, i, h, v] = yawr*rhow*g*breadth**2

    f.close()

    # Transform from VERES frame (x to stern, z up, y stbd) to 
    # MCSimPython frame (x forward, z down, y stbd)
    # Both are right hand coordinate systems (only pi rotated about y.)

    T = J([0., 0., 0., 0., np.pi, 0.])@np.array([1., 1., 1., 1., 1., 1.])
    T = np.array([-1, 1, -1, -1, 1, -1])

    # Flip the order of headings to correspond to MCSimPython
    # heading convention. (Head sea : beta = 180 deg.)
    # VERES convention is (Head sea : beta = 0 deg.)

    drift_frc_n = np.flip(drift_frc, axis=2)

    for i in range(6):
        drift_frc_n[i] = T[i]*drift_frc_n[i]

    return drift_frc_n


def plot_raos(raos, freqs, plot_polar=True, rao_type="motion", wave_angle=0, figsize=(16, 8)):
    """Plot the force or motion RAOs. 

    The RAOs should be complex. 

    Parameters
    ----------
    raos : array_like
        Array of motion RAOs for k dofs, n freqs, m headings, and j velocities.
    freqs : array_like
        Array of frequencies for which the RAOs are computed.
    rao_type : string (default = "motion")
        Type of RAO. Either "motion" or "force". Defaults to "motion"
    plot_polar : bool (default = True)
        Either plot the RAOs in polar- or cartesian coordinates. Defaults to polar.
    wave_angle : int (default = 0)
        Index of the relative incident wave anlge. Defaults to 0, which should
        correspond to following sea.
    figsize : tuple (default = (16, 8))
        Size of the plot. Defaults to (16, 8).  
    
    """
    titles = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
    if plot_polar:
        fig, axs = plt.subplots(2, 3, figsize=figsize, constrained_layout=True, subplot_kw={'projection': 'polar'})
        for i in range(6):
            plt.sca(axs[i//3, i%3])
            plt.title(f"RAO {titles[i]}")
            plt.plot(np.angle(raos[i, :, wave_angle, 0]), np.abs(raos[i, :, wave_angle, 0]))
            plt.plot(np.angle(raos[i, 0, wave_angle, 0]), np.abs(raos[i, 0, wave_angle, 0]), 'ro', label="$\omega_{min}$")
            plt.plot(np.angle(raos[i, -1, wave_angle, 0]), np.abs(raos[i, -1, wave_angle, 0]), 'go', label="$\omega_{max}$")
            if (i < 3) and (rao_type == "motion"):
                plt.gca().set_rmax(1)
            plt.legend()

    if not plot_polar:
        fig, axs = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        fig.suptitle("RAO Amplitude")
        for i in range(6):
            plt.sca(axs[i//3, i%3])
            plt.plot(freqs, np.abs(raos[i, :, wave_angle, 0]))
            plt.xlabel("$\omega \; [rad/s]$")
            if i < 3:
                plt.ylabel(r"$\frac{\eta}{\zeta_a} \; [\frac{m}{m}]$")
            else:
                plt.ylabel(r"$\frac{\eta}{\zeta_a} \; [\frac{rad}{m}]$")
        
        fig, axs = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        for i in range(6):
            plt.sca(axs[i//3, i%3])
            plt.plot(freqs, np.angle(raos[i, :, wave_angle, 0]))
            plt.xlabel("$\omega \; [rad/s]$")
            plt.ylabel("$\phi \; [rad]$")
    
    plt.show()

def _complete_sector_coeffs(vessel_config: dict):
    freqs = np.asarray(vessel_config['freqs'])
    vel = np.asarray(vessel_config['velocity'])
    drift_coeffs = np.asarray(vessel_config['driftfrc']['amp'])
    headings = np.asarray(vessel_config['headings'])
    forceRAOc = np.asarray(vessel_config['forceRAO']['complex'])
    forceRAOamp = np.asarray(vessel_config['forceRAO']['amp'])
    forceRAOphase = np.asarray(vessel_config['forceRAO']['phase'])
    motionRAOc = np.asarray(vessel_config['motionRAO']['complex'])
    motionRAOamp = np.asarray(vessel_config['motionRAO']['amp'])
    motionRAOphase = np.asarray(vessel_config['motionRAO']['phase'])

    heading_new = np.deg2rad(np.arange(0, 360, 10))
    forceRAOc_new = np.zeros((6, freqs.size, heading_new.size, vel.size), dtype=np.complex128)
    forceRAOamp_new = np.zeros_like(forceRAOc_new, dtype=np.float64)
    forceRAOphase_new = np.zeros_like(forceRAOc_new, dtype=np.float64)
    motionRAOc_new = np.zeros_like(forceRAOc_new, dtype=np.complex128)
    motionRAOamp_new = np.zeros_like(forceRAOc_new, dtype=np.float64)
    motionRAOphase_new = np.zeros_like(forceRAOc_new, dtype=np.float64)

    drift_coeffs_new = np.zeros((6, freqs.size, heading_new.size, vel.size))

    # Copy the drift coeffs
    drift_coeffs_new[:, :, :19, :] = drift_coeffs
    drift_coeffs_new[0, :, 19:] = np.copy(drift_coeffs[0, :, -2:0:-1])
    drift_coeffs_new[1, :, 19:] = -np.copy(drift_coeffs[1, :, -2:0:-1])
    drift_coeffs_new[2, :, 19:] = -np.copy(drift_coeffs[2, :, -2:0:-1])

    new_raos = [forceRAOc_new, forceRAOamp_new, forceRAOphase_new, motionRAOc_new, motionRAOamp_new, motionRAOphase_new]
    old_raos = [forceRAOc, forceRAOamp, forceRAOphase, motionRAOc, motionRAOamp, motionRAOphase]
    for i in range(len(new_raos)):
        new_raos[i][:, :, :19] = old_raos[i]
        new_raos[i][:, :, 19:] = old_raos[i][:, :, -2:0:-1]
    
    # T = np.array([-1, 1, -1, -1, 1, -1])
    
    for dof in [1, 3, 5]:
        forceRAOc_new[dof, :, 19:] = np.conjugate(forceRAOc_new[dof, :, 19:])
        motionRAOc_new[dof, :, 19:] = np.conjugate(motionRAOc_new[dof, :, 19:])
        forceRAOphase_new[dof, :, 19:] *= -1
        motionRAOphase_new[dof, :, 19:] *= -1
    
    vessel_config['headings'] = heading_new.tolist()
    vessel_config['motionRAO']['complex'] = motionRAOc_new.tolist()
    vessel_config['motionRAO']['amp'] = motionRAOamp_new.tolist()
    vessel_config['motionRAO']['phase'] = motionRAOphase_new.tolist()
    vessel_config['forceRAO']['complex'] = forceRAOc_new.tolist()
    vessel_config['forceRAO']['amp'] = forceRAOamp_new.tolist()
    vessel_config['forceRAO']['phase'] = forceRAOphase_new.tolist()
    vessel_config['driftfrc']['amp'] = drift_coeffs_new.tolist()

def generate_config_file(input_files_paths: list = None, input_file_dir: str = None):
    """Generate a .json configuration file for a vessel. The function can take
    either a list of file locations, or the path to the directory containing the
    result files. 
    
    Parameters
    ----------
    input_files_paths: list (default = None)
        List of paths for each input file (result file) from VERES. Should contain
        .re1, .re2, .re7, and .re8. Defaults to None.
    input_file_dir : str (default = None)
        Path to directory containing VERES result files. The folder should contain
        .re1, .re2, .re7, and .re8. Default to None.
    
    See also
    --------
    utils.read_tf()
    utils.read_wave_drift()
    """

    # Verify that the all necessary input files are given.
    # need .re1, .re2, .re7, and .re8
    # - .re1 motion rao
    # - .re2 added resistance
    # - .re7 hydrod coeffs
    # - .re8 force rao

    file_type_requirm = ['.re1', '.re2', '.re7', '.re8']
    if input_files_paths is not None:
        input_file_types = [file[-4:] for file in input_files_paths]
        if not sorted(input_file_types)==sorted(file_type_requirm):
            raise ValueError(f"The correct files are not distributed. Want: {file_type_requirm} files.")
        else:
            re1, re2, re7, re8 = input_files_paths
    elif input_file_dir is not None:
        input_file_types = [file[-4:] for file in os.listdir(input_file_dir) if 're' in file.split('.')[-1]]
        print(sorted([os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir) if "re" in file.split('.')[-1]]))
        re1, re2, re7, re8, *_ = sorted([os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir) if "re" in file.split('.')[-1]])
    else:
        raise FileNotFoundError("Could not find the result files from VERES.")
    
    vessel_config = {}
    vessel_config['motionRAO'] = {}
    vessel_config['forceRAO'] = {}
    vessel_config['driftfrc'] = {}
    vessel_config['Bv44'] = {}
    
    # Compute RAOs.
    print("Read input files".center(100, '-'))
    print("Read motion RAOs (.re1)...")
    freqs, headings, vels, motion_rao_c, motion_rao_amp, motion_rao_phase = read_tf(re1)
    print("Read force RAOs (.re8)...")
    _, _, _, force_rao_c, force_rao_amp, force_rao_phase = read_tf(re8)
    print("Read wave drift data (.re2)...")
    drift_frc = read_wave_drift(re2)
    print("Read hydrodynamic parameters (.re7)...")
    Mrb, A, B, C, Bv44_lin, Bv44_nonlin, Bv44_linearized = read_hydrod(re7)
    print("COMPLETE".center(100, '.'))
    # Functions for reading the other files
        ## ADD FUNCTIONS HERE
    # Add the inputs to dictionary.
    vessel_config['freqs'] = freqs.tolist()
    vessel_config['headings'] = headings.tolist()
    vessel_config['velocity'] = vels.tolist()
    vessel_config['motionRAO']['complex'] = motion_rao_c.tolist()
    vessel_config['motionRAO']['amp'] = motion_rao_amp.tolist()
    vessel_config['motionRAO']['phase'] = motion_rao_phase.tolist()
    vessel_config['motionRAO']['w'] = freqs.tolist()
    vessel_config['forceRAO']['complex'] = force_rao_c.tolist()
    vessel_config['forceRAO']['amp'] = force_rao_amp.tolist()
    vessel_config['forceRAO']['phase'] = force_rao_phase.tolist()
    vessel_config['forceRAO']['w'] = freqs.tolist()
    vessel_config['driftfrc']['amp'] = drift_frc.tolist()
    vessel_config['A'] = A.tolist()
    vessel_config['B'] = B.tolist()
    vessel_config['C'] = C.tolist()
    vessel_config['MRB'] = Mrb.tolist()
    vessel_config['Bv44']['Linear'] = Bv44_lin.tolist()
    vessel_config['Bv44']['Nonlin'] = Bv44_nonlin.tolist()
    vessel_config['Bv44']['Linearized'] = Bv44_linearized.tolist()
    Bv = np.zeros((6, 6))
    # Bv[3, 3] = Bv44_linearized
    vessel_config['Bv'] = np.zeros((6, 6)).tolist()   # Should add some viscous damping estimates here.

    if headings.size <= 19:
        print(f"VERES results only for heading {np.min(headings)} to {np.max(headings)}. ")
        _complete_sector_coeffs(vessel_config)
    elif np.max(headings) <= np.pi:
        print(f"VERES results only for headings {headings}.")
        raise NotImplementedError
    
    
    return vessel_config
    # Add some saving method here.
    
    