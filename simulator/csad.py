# csad.py
import numpy as np
import os
import json

from simulator.vessel import Vessel
from utils import timeit, J, Smat, Rzyx, pipi


"""
Vessel models for C/S Arctic Drillship.

Data for CSAD Vessel is found in vessel_data/CSAD/

Models:
- 3DOF Maneuvering model
"""

DATA_DIR = os.path.abspath(os.path.join(os.path.join(
    os.getcwd(), os.pardir), 'vessel_data', 'CSAD'))


class CSADMan3DOF(Vessel):

    def __init__(self, dt, *args, dof=3, config_file="csad_params.json", **kwargs):
        config_file = os.path.join(DATA_DIR, config_file)
        super().__init__(dt, config_file=config_file, dof=dof)

    def x_dot(self, *args, **kwargs):
        pass


class CSAD6DOF(Vessel):
    """
    6 DOF stationkeeping model. Unified seakeeping and maneuvering model
    with fluid memory effects. 
    """

    def __init__(self, dt, *args, config_file="vessel_json.json", vessel_abc="vesselABC_json.json", **kwargs):
        with open(os.path.join(DATA_DIR, config_file)) as f:
            vessel = json.load(f)
        with open(os.path.join(DATA_DIR, vessel_abc)) as ff:
            vesselABC = json.load(ff)
        self._dof = 6
        self._Mrb = np.asarray(vesselABC['MRB'])
        self._Ma = np.asarray(vesselABC['MA'])
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)
        self._D = vessel['Bv']
        self._G = np.asarray(vesselABC['G'])
        self._x = np.zeros(2*self._dof)
        self._x_dot = np.zeros_like(self._x)
        self._eta = np.zeros(self._dof)
        self._nu = np.zeros(self._dof)
        self._dt = dt

        # Set up state-space models for fluid memory effects. 
        # Fluid memory surge (no coupling for csad)
        self._Ar_surge = np.asarray(vesselABC['Ar'][0])
        self._Br_surge = np.asarray(vesselABC['Br'][0])
        self._Cr_surge = np.asarray(vesselABC['Cr'][0])
        self._xr_surge = np.zeros(self._Ar_surge.shape[0])
        self._dot_xr_sway = self._xr_surge

        # Fluid memory sway (sway-roll-yaw coupling)
        self._Ar_sway = [
            np.asarray(vesselABC['Ar'][7]),
            np.asarray(vesselABC['Ar'][9]),
            np.asarray(vesselABC['Ar'][11])
        ]
        self._Br_sway = [np.asarray(vesselABC['Br'][i]) for i in [7, 9, 11]]
        self._Cr_sway = [np.asarray(vesselABC['Cr'][i]) for i in [7, 9, 11]]
        self._xr_sway = [np.zeros(self._Ar_sway[i].shape[0]) for i in range(3)]
        self._dot_xr_sway = self._xr_sway

        # Fluid memory heave (heave-pich coupling)
        self._Ar_heave = [
            np.asarray(vesselABC['Ar'][14]),
            np.asarray(vesselABC['Ar'][16]),
        ]
        self._Br_heave = [np.asarray(vesselABC['Br'][i]) for i in [14, 16]]
        self._Cr_heave = [np.asarray(vesselABC['Cr'][i]) for i in [14, 16]]
        self._xr_heave = [np.zeros(self._Ar_heave[i].shape[0]) for i in range(2)]
        self._dot_xr_heave = self._xr_heave

        # Fluid memory roll (sway-roll-yaw coupling)
        self._Ar_roll = [
            np.asarray(vesselABC['Ar'][19]),
            np.asarray(vesselABC['Ar'][21]),
            np.asarray(vesselABC['Ar'][23])
        ]
        self._Br_roll = [np.asarray(vesselABC['Br'][i]) for i in [19, 21, 23]]
        self._Cr_roll = [np.asarray(vesselABC['Cr'][i]) for i in [19, 21, 23]]
        self._xr_roll = [np.zeros(self._Ar_roll[i].shape[0]) for i in range(3)]
        self._dot_xr_roll = self._xr_roll

        # Fluid memory pitch (heave-pitch coupling)
        self._Ar_pitch = [
            np.asarray(vesselABC['Ar'][26]),
            np.asarray(vesselABC['Ar'][28]),
        ]
        self._Br_pitch = [np.asarray(vesselABC['Br'][i]) for i in [26, 28]]
        self._Cr_pitch = [np.asarray(vesselABC['Cr'][i]) for i in [26, 28]]
        self._xr_pitch = [np.zeros(self._Ar_pitch[i].shape[0]) for i in range(2)]
        self._dot_xr_pitch = self._xr_pitch

        # Fluid memory yaw (sway-roll-yaw coupling)
        self._Ar_yaw = [
            np.asarray(vesselABC['Ar'][31]),
            np.asarray(vesselABC['Ar'][33]),
            np.asarray(vesselABC['Ar'][35])
        ]
        self._Br_yaw = [np.asarray(vesselABC['Br'][i]) for i in [31, 33, 35]]
        self._Cr_yaw = [np.asarray(vesselABC['Cr'][i]) for i in [31, 33, 35]]
        self._xr_yaw = [np.zeros(self._Ar_yaw[i].shape[0]) for i in range(3)]
        self._dot_xr_yaw = self._xr_yaw


    def x_dot(self, Uc, betac, tau, *args, **kwargs):
        nu_cn = Uc*np.array([np.cos(betac), np.sin(betac), 0, 0, 0, 0])
        nu_c = J(*pipi(self._eta[3:])).T@nu_cn
        print(f"nu_c[4]: {nu_c[4]}")
        # nu_c = np.concatenate([nu_c3, np.zeros(3)])
        nu_r = self._nu - nu_c
        print(f"nu_r[4]: {nu_r[4]}")
        S6 = np.block([
            [-Smat(self._nu[3:]), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3))]
        ])
        dnu_c = S6@nu_c
        

        mew = self.memory_effects(nu_r)
        #mew = np.zeros(6) # The memory effects goes to infinite

        self._eta_dot = J(*pipi(self._eta[3:]))@self._nu
        self._nu_dot = self._Minv@(tau - self._D@nu_r - self._G@self._eta + self._Ma@dnu_c)
        self._x_dot = np.concatenate([self._eta_dot, self._nu_dot])
    

    @timeit
    def memory_effects(self, nu, *args, **kwargs):
        # For surge: velocities = nu[0]
        # For sway, roll and yaw: velocities = nu[1], nu[3], nu[5]
        # For heave and pitch: velocities = nu[2], nu[4]
        print(nu)
        u1 = nu[0]
        u2 = nu[[1, 3, 5]]
        print(u2)
        u3 = nu[[2, 4]]
        u4 = u2
        u5 = u3
        u6 = u2
        
        mew = np.zeros(self._dof)
        # Surge
        self._dot_xr_surge = self._Ar_surge@self._xr_surge + self._Br_surge*u1
        self._xr_surge += self._dt * self._dot_xr_surge
        mew[0] = self._Cr_surge@self._xr_surge
        
        # Sway
        for i in range(len(self._Ar_sway)):
            self._dot_xr_sway[i] = self._Ar_sway[i]@self._xr_sway[i] + self._Br_sway[i]*u2[i]
            self._xr_sway[i] += self._dt * self._dot_xr_sway[i]
            mew[1] += self._Cr_sway[i]@self._xr_sway[i]

        # Heave
        for i in range(len(self._Ar_heave)):
            self._dot_xr_heave[i] = self._Ar_heave[i]@self._xr_heave[i] + self._Br_heave[i]*u3[i]
            self._xr_heave[i] += self._dt * self._dot_xr_heave[i]
            mew[2] += self._Cr_heave[i]@self._xr_heave[i]

        # Roll
        for i in range(len(self._Ar_roll)):
            self._dot_xr_roll[i] = self._Ar_roll[i]@self._xr_roll[i] + self._Br_roll[i]*u4[i]
            self._xr_roll[i] += self._dt * self._dot_xr_roll[i]
            mew[3] += self._Cr_roll[i]@self._xr_roll[i]

        # Pitch
        for i in range(len(self._Ar_pitch)):
            self._dot_xr_pitch[i] = self._Ar_pitch[i]@self._xr_pitch[i] + self._Br_pitch[i]*u5[i]
            self._xr_pitch[i] += self._dt * self._dot_xr_pitch[i]
            mew[4] += self._Cr_pitch[i]@self._xr_pitch[i]
        
        # Yaw
        for i in range(len(self._Ar_yaw)):
            self._dot_xr_yaw[i] = self._Ar_yaw[i]@self._xr_yaw[i] + self._Br_yaw[i]*u6[i]
            self._xr_yaw[i] += self._dt * self._dot_xr_yaw[i]
            mew[5] += self._Cr_yaw[i]@self._xr_yaw[i]

        return mew
        


if __name__ == "__main__":
    vessel = CSADMan3DOF(0.1)

