# Module for computing wave loads on vessel
import numpy as np
import json
from utils import timeit

class WaveLoad:
    """
    Class for computation of wave loads on floating structure. Both 
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
    _params : 
        Vessel parameters with 1nd and 2nd order transfer functions.
    _g : float
        Gravitational acceleration constant.
    _rho : float
        Water density.
    _force_rao_amp : array_like
        Vessel force RAO amplitude for the given wave frequencies _freq.
    _force_rao_phase : array_like
        Vessel force RAO phase for the given wave frequencies _freq.
    """

    def __init__(self, wave_amps, freqs, eps, angles, config_file, rho=1025, g=9.81, dof=6):
        with open(config_file, 'r') as f:
            vessel_params = json.load(f)
        self._N = wave_amps.shape[0]
        self._amp = wave_amps
        self._freqs = freqs
        self._eps = eps
        self._angles = angles
        self._qtf_angles = np.asarray(vessel_params['headings'])
        self._params = vessel_params
        # self._force_rao_amp, self._force_rao_phase = self._set_force_raos()
        self._g = g
        self._k = freqs**2/g
        self._rho = rho
        self._W = freqs[:, None] - freqs
        self._P = eps[:, None] - eps
        self._Q = self._full_qtf_6dof(
            np.asarray(vessel_params['headings']),
            np.asarray(vessel_params['freqs']),
            np.asarray(vessel_params['driftfrc']['amp'])[:, :, :, 0]
        )
        # self._forceRAOamp = np.asarray(vessel_params['forceRAO']['amp'])[:, :, :, 0]
        # self._forceRAOphase = np.asarray(vessel_params['forceRAO']['phase'])[:, :, :, 0]
        self._set_force_raos()


    def _set_force_raos(self):
        """
        Function to set the force raos to be used in calculation
        of 1st and second order wave loads. 
        Selected for the wave frequencies of the sea state by closest index.
        """
        amp = np.array(self._params['forceRAO']['amp'])[:, :, :, 0]
        phase = np.array(self._params['forceRAO']['phase'])[:, :, :, 0]
        freqs = np.array(self._params['freqs'])

        # DO WE NEED TO SET THE HEADING INDEX FOR THIS FUNCTION?

        heading_indx = np.array([np.argmin(np.abs(self._qtf_angles - angle)) for angle in self._angles])
        freq_indx = np.array([np.argmin(np.abs(freqs - w)) for w in self._freqs])


        self._forceRAOamp = np.zeros((6, self._N, len(self._qtf_angles)))
        self._forceRAOphase = np.zeros((6, self._N, len(self._qtf_angles)))
        for dof in range(6):
            self._forceRAOamp[dof] = amp[dof, [freq_indx], :][0]
            self._forceRAOphase[dof] = phase[dof, [freq_indx], :][0]

    @timeit
    def first_order_loads(self, t, rel_angle, eta):
        """
        Calculate first order wave-loads by super position of 
        wave load from each individual wave component.

        (Assumption: The force RAO amplitude and phase is known).

        """
        # CHECK THE ANGLE DEFINITION FOR RELATIVE WAVE ANGLE.
        # ONLY COMPUTING FOR UNI-DIR WAVES KNOW.
        heading_index = np.argmin(np.abs(self._qtf_angles - np.abs(rel_angle)))

        rao_amp = self._forceRAOamp[:, :, heading_index]
        rao_phase = self._forceRAOphase[:, :, heading_index]
    
        x = eta[0]
        y = eta[1]

        rao = rao_amp*np.cos(
            self._freqs*t -
            self._k*x*np.cos(self._angles) - self._k*y*np.sin(self._angles) -
            self._eps - rao_phase
        )
        tau_wf = np.zeros(6)

        # for dof in range(6):
        #     tau_wf[dof] = self._amp@rao[dof]

        tau_wf = rao@self._amp

        return tau_wf

    @timeit
    def second_order_loads(self, t, rel_angle):
        """
        Calcualation of second order drift loads.

        Estimate 2nd order drift loads using either Newman or formulation from 
        Standing, Brendling and Wilson (as used by OrcaFlex).

        References
        ----------
        [Newman 1974] INSERT HER
        [Standing, Brendling and Wilson]
        """
        # Implementation - currently for only 1 DOF

        # Get the QTF matrix for the given heading.
        heading_index = np.argmin(np.abs(self._qtf_angles - np.abs(rel_angle)))
        Q = self._Q[:, heading_index, :, :]

        tau_sv = np.zeros(6)
        for i in [0, 1, 5]:
            tau_sv[i] = np.real(self._amp.T@ (Q[i]*np.exp(self._W*(1j*t) + 1j*self._P)) @ self._amp)
        return tau_sv


    @timeit
    def _full_qtf(self, qtf_headings, qtf_freqs, qtfs, angle, method="Newman"):
        """
        Calculate the full QTF matrix for a fixed heading by Newmans approximation or 
        the geometric mean formulation. 
        
        #### MAKE THIS RETURN A (36, 47, 47) MATRIX AS - ONE 47 x 47 MATRIX FOR EACH HEADING!!!

        Note
        ----
        Only implemented for unidirectional waves at this moment. 

        Parameters
        ----------
        qtf_headings : 1D-array
            Headings used in calculation of QTFs
        qtf_freqs : 1D-array
            Frequencies used in calculation of QTFs
        qtf_diag : 1D-array
            The diagonal terms of the QTF for a given heading.
        angle : 1D-array
            Relative wave directional anlge.
        method : string (default='Newman')
            Method to be used for approximation.
        
        Returns
        -------
        QTF : 2D-array
            Full QTF matrix for a fixed heading.

        References
        ----------
        [Newman, 1974] INSERT HER
        [Standing, Brendling and Wilson]
        """

        # First -> pick the values to be used for a given angle.
        heading_index = np.argmin(np.abs(qtf_headings - angle))
        qtf_diag = qtfs[:, heading_index, 0]

        # Second -> interpolate to the frequencies used in wave spectra. (self._freqs)
        # Chosen to use closes frequency instead - this could be changed to an interpolation scheme instead.
        freq_indices = [np.argmin(np.abs(qtf_freqs - wave_freq)) for wave_freq in self._freqs]

        # Third -> generate full QTF matrix based on either Newman approx or Geometric Mean

        QTF = np.empty((self._N, self._N))
        for i, freq_i in enumerate(freq_indices):
            Q_i = qtf_diag[freq_i]
            for j, freq_j in enumerate(freq_indices):
                Q_j = qtf_diag[freq_j]

                if method=="Newman":
                    QTF[i, j] = 0.5*(Q_i + Q_j)
                else:
                    # Using geometric mean approximation
                    sgn_equal = np.sign(Q_i) == np.sign(Q_j)
                    if sgn_equal:
                        QTF[i, j] = np.sign(Q_i)*np.abs(Q_i*Q_j)**(.5)
                    else:
                        QTF[i, j] = 0

        return QTF


    @timeit
    def _full_qtf_6dof(self, qtf_headings, qtf_freqs, qtfs, method="Newman"):
        """
        Generate the full QTF matrix for all DOF, all headings with calculated QTF
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
        print(len(freq_indices))
        Q = np.zeros((6, len(qtf_headings), self._N, self._N))

        for dof in range(6):
            Qdiag = qtfs[dof, [freq_indices], :].copy()
            for i in range(len(qtf_headings)):
                if method == "Newman":
                    Q[dof, i] = 0.5*(Qdiag[0, :, i, None] + Qdiag[0, :, i])
        
        # From config file, qtf[2] is yaw - not heave. Changing this here.
        temp = Q[2]
        Q[5] = Q[2]
        Q[2] = np.zeros_like(Q[0])
        return Q
        