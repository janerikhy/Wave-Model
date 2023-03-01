# Module for computing wave loads on vessel

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2022-10-26
# Revised: 2023-01-30 Jan-Erik Hygen    Add multidir wave loads.
#          2023-02-01 Jan-Erik Hygen    Add finite depth dispersion relation.
#          2023-02-05 Jan-Erik Hygen    Add relative incident wave angle
#                                       computation function.
#          2023-02-07 Jan-Erik Hygen    Add interpolation of force RAO.
# 
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import interp1d
import json
from MCSimPython.utils import to_positive_angle, pipi

class WaveLoad:
    """Class for computation of wave loads on floating structure. Both 
    1st and 2nd order wave loads are considered.

    Attributes
    ----------
    _N : int
        Number of wave components.
    _amp : 1D-array
        Wave amplitudes.
    _freqs : 1D-array
        Wave frequencies.
    _eps : 1D-array
        Random wave phases.
    _angles : 1D-array
        Wave directional angles.
    _qtf_angles: 1D-array
        Array of angles used in computation of QTFs in Veres.
    _params : dict
        Vessel parameters with 1nd and 2nd order transfer functions.
    _g : float
        Gravitational acceleration constant.
    _k : 1D-array
        Wave number for each wave component found by linear deep-water dispersion relation.
    _rho : float
        Water density.
    _W : N x N - array
        2D array of difference frequencies for each wave pair.
    _P : N x N - array
        2D array of phase difference for each wave pair.
    _Q : 6 x M x N x N - array
        Array of full QTFs for each DOF, for M relative wave angles and N wave components.
    _forceRAOamp : array_like
        Vessel force RAO amplitude for the given wave frequencies _freq.
    _forceRAOphase : array_like
        Vessel force RAO phase for the given wave frequencies _freq.

    Methods
    -------
    _set_force_raos()
    _first_order_loads()
    _second_order_loads()
    _full_qtf_6dof()
    """

    def __init__(self, 
                wave_amps,
                freqs, 
                eps, 
                angles, 
                config_file, 
                rho=1025, 
                g=9.81,
                dof=6,
                depth = 100,
                deep_water=True,
                interpolate=True):
        with open(config_file, 'r') as f:
            vessel_params = json.load(f)
        self._N = wave_amps.shape[0]
        self._amp = wave_amps
        self._freqs = freqs
        self._eps = eps
        self._angles = angles
        self._depth = depth
        self._qtf_angles = np.asarray(vessel_params['headings'])
        self._params = vessel_params
        self._g = g
        self._k = freqs**2/g
        if not deep_water:
            self._k = self.wave_number(self._freqs)
        self._rho = rho
        self._W = freqs[:, None] - freqs
        self._P = eps[:, None] - eps
        self._Q = self._full_qtf_6dof(
            np.asarray(vessel_params['headings']),
            np.asarray(vessel_params['freqs']),
            np.asarray(vessel_params['driftfrc']['amp'])[:, :, :, 0]
        )
        self._set_force_raos(interpolate)

    def __call__(self, time, eta):
        """Calculate first- and second-order wave loads."""
        tau_wf = self.first_order_loads(time, eta)
        tau_sv = self.second_order_loads(time, eta[-2])
        return tau_wf + tau_sv

    def _set_force_raos(self, interpolate=None):
        """
        Function to set the force RAOs to be used in calculation
        of 1st order wave loads.
        Selected for the wave frequencies of the sea-state by closest index.
        """
        amp = np.array(self._params['forceRAO']['amp'])[:, :, :, 0]
        phase = np.array(self._params['forceRAO']['phase'])[:, :, :, 0]
        freqs = np.array(self._params['freqs'])

        if interpolate:
            f1 = interp1d(freqs, np.abs(amp), axis=1, bounds_error=False, fill_value=(amp[:,0,:], 0))
            f2 = interp1d(freqs, phase, axis=1, bounds_error=False, fill_value=(0, phase[:,-1,:]))
            self._forceRAOamp = f1(self._freqs)
            self._forceRAOphase = f2(self._freqs)
        else:
            freq_indx = np.array([np.argmin(np.abs(freqs - w)) for w in self._freqs])

            self._forceRAOamp = np.zeros((6, self._N, len(self._qtf_angles)))
            self._forceRAOphase = np.zeros((6, self._N, len(self._qtf_angles)))
            for dof in range(6):
                self._forceRAOamp[dof] = amp[dof, [freq_indx], :][0]
                self._forceRAOphase[dof] = phase[dof, [freq_indx], :][0]

    def first_order_loads(self, t, eta):
        """
        Calculate first order wave-loads by super position of 
        wave load from each individual wave component.

        (Assumption: The force RAO amplitude and phase is known).

        Parameters
        ----------
        self : Vessel object
        t : float
            Time instance
        rel_angle : array_like
            Relative wave incident angle for each wave component
        eta : array_like
            Pose of vessel

        Return
        ------
        tau_wf : 6x1 array
            First order wave-frequency load for each DOF.
        """
        rel_angle = self._relative_incident_angle(eta[-1])

        rao_amp, rao_phase = self._rao_interp(rel_angle)
    
        # Horizontal position of vessel (N, E).
        x = eta[0]
        y = eta[1]

        rao = rao_amp*np.cos(
            self._freqs*t -
            self._k*x*np.cos(self._angles) - self._k*y*np.sin(self._angles) -
            self._eps - rao_phase
        )

        tau_wf = rao@self._amp

        return tau_wf

    def second_order_loads(self, t, heading):
        """
        Calcualation of second order drift loads.

        Estimate 2nd order drift loads using either Newman or formulation from 
        Standing, Brendling and Wilson (as used by OrcaFlex).

        Parameters:
        -----------
        t : float
            Time.
        heading: float
            Vessel heading in NED-frame.

        Return:
        -------
        tau_sv : 6x1 - array
            Array of slowly-vayring load components for each DOF.

        References
        ----------
        .. [1] J. N. Newman. Marine hydrodynamics / J. N. Newman. MIT Press Cambridge, Mass, 1977. ISBN
           0262140268.
        """

        # Use the mean relative angle to compute the slowly-varying loads
        rel_angle = np.mean(self._relative_incident_angle(heading))

        # Get the QTF matrix for the given heading for each DOF.
        heading_index = np.argmin(np.abs(self._qtf_angles - rel_angle))
        Q = self._Q[:, heading_index, :, :]

        tau_sv = np.real(self._amp@(Q*np.exp(self._W*(1j*t) + 1j*self._P))@self._amp)
        
        return tau_sv

    def _full_qtf_6dof(self, qtf_headings, qtf_freqs, qtfs, method="Newman"):
        """Generate the full QTF matrix for all DOF, all headings with calculated QTF
        and for all wave frequency components.

        VERES ONLY CALCULATES DRIFT FORCES FROM SURGE SWAY AND YAW. THEREFOR THE REMAINING
        DOFs WILL NOT REQUIRE QTF AND WILL BE SET TO ZERO.

        Parameters
        ----------
        qtf_headings : 1D-array
            Headings used for calculations of QTFs in diffraction program (ShipX etc)
        qtf_freqs : 1D-array
            Frequencies used in calculations of QTS in diffraction program (Shipx etc)
        qtfs : 6 x n x m - array
            QTFs calculated for 6 DOF for n frequencies and m headings.
        method : string (default="Newman")
            Method to be used for approximating off-diagonal terms. Newman approximation
            is the default method, geometric mean (as used in OrcaFlex) is the other.
        
        Returns
        -------
        Q : 6 x M x N x N array
            Approximated full QTF for 6 DOF, M headings and N wave frequencies.
        """
        
        freq_indices = [np.argmin(np.abs(qtf_freqs - freq)) for freq in self._freqs]
        Q = np.zeros((6, len(qtf_headings), self._N, self._N))

        for dof in range(6):
            Qdiag = qtfs[dof, [freq_indices], :].copy()
            for i in range(len(qtf_headings)):
                if method == "Newman":
                    Q[dof, i] = 0.5*(Qdiag[0, :, i, None] + Qdiag[0, :, i])
        
        # From config file, qtf[2] is yaw - not heave. Changing this here.
        Q[5] = Q[2].copy()
        Q[2] = np.zeros_like(Q[0])
        return Q
        
    def wave_number(self, omega, tol=1e-5):
        """Wave number of wave components.
        Calculate the wave number using the dispersion relation. The wave number
        is calculated through an iterative scheme. 

        Parameters:
        -----------
        omega : array_like
            1D array of wave frequencies in rad.
        tol : float
            Tolerance used in the iteration to define when the solution converge.

        Returns:
        --------
        k : array_like
            1D array of wave numbers.
        """
        k = np.zeros(self._N)
        for i, w in enumerate(omega):
            k_old = w**2/self._g
            k_new = w**2/(self._g * np.tanh(k_old * self._depth))
            diff = np.abs(k_old - k_new)
            count = 0
            while diff > tol:
                k_old = k_new
                k_new = w**2/(self._g * np.tanh(k_old * self._depth))
                diff = np.abs(k_old - k_new)
                count += 1
            k[i] = k_new
            print(count)
        return k

    def _relative_incident_angle(self, heading):
        """The relative wave incident angle gamma.

        Calculate the relative angle for each wave component. 
        Gamma = 0.0 is defined as following sea, gamma = 180 as head sea, and gamma = 90 as beam sea.

        Parameters:
        -----------
        
        heading : float
            Heading of the vessel
        
        Returns:
        --------
        gamma : array_like
            Array of relative incident wave angles.
        """
        return to_positive_angle(pipi(self._angles - heading))

    def _rao_interp(self, rel_angle):
        index_lb = np.argmin(np.abs(np.rad2deg(self._qtf_angles) - np.floor(np.rad2deg(rel_angle[:, None])/10)*10), axis=1)
        index_ub = np.where(index_lb < 35, index_lb + 1, 0)

        freq_ind = np.arange(0, self._N)
        
        rao_amp_lb = self._forceRAOamp[:, freq_ind, index_lb]
        rao_phase_lb = self._forceRAOphase[:, freq_ind, index_lb]

        rao_amp_ub = self._forceRAOamp[:, freq_ind, index_ub]
        rao_phase_ub = self._forceRAOphase[:, freq_ind, index_ub]

        theta1, theta2 = self._qtf_angles[index_lb], self._qtf_angles[index_ub]
        scale = pipi(rel_angle - theta1)/pipi(theta2 - theta1)
        # Ensure that diff theta1 and theta2 is the smallest signed angle
        rao_amp = rao_amp_lb + (rao_amp_ub - rao_amp_lb)*scale
        rao_phase = rao_phase_lb + (rao_phase_ub - rao_phase_lb)*scale
        return rao_amp, rao_phase
