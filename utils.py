# Utility functions
import numpy as np
from time import time


def complex_to_polar(complex_values):
    complex_values = np.asarray_chkfinite(complex_values)
    amp = np.abs(complex_values)
    theta = np.angle(complex_values)
    return amp, theta


def polar_to_complex(amp, theta):
    amp = np.asarray_chkfinite(amp)
    theta = np.asarray_chkfinite(theta)
    return amp*(np.cos(theta) + 1j * np.sin(theta))


def pipi(theta):
    return np.mod(theta + np.pi, 2*np.pi) - np.pi


def Rx(phi):
    return np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])


def Ry(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def Rz(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])


def Rzyx(eta):
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


def timeit(func):
    """
    Decorator for measuring execution time of function.
    """
    def wrapper(*args, **kwargs):
        t1 = time()
        results = func(*args, **kwargs)
        t2 = time()
        print(f"Execution time of {func.__name__}: {(t2 - t1):.4f}")
        return results
    return wrapper