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
