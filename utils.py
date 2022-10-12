# Utility functions
import numpy as np


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
