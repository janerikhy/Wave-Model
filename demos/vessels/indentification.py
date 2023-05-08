from MCSimPython.simulator import RVG_DP_6DOF
from MCSimPython.utils import system_identification as sys_id

import os
import json
import numpy as np
import matplotlib.pyplot as plt

vessel = RVG_DP_6DOF(0.1, config_file="rvg3_config.json")
with open(vessel._config_file, 'r') as f:
    config = json.load(f)

A = np.asarray(config['A'])[:, :, :, 0]
B = np.asarray(config['B'])[:, :, :, 0]
w = np.asarray(config['freqs'])

MA, Ar, Br, Cr = sys_id(w, A, B, max_order=7, method=2, plot_estimate=True)
with np.printoptions(precision=3, suppress=True):
    print(MA)
    
    
# Save the data to a configuration file to use it for simulation.
file_name = "rvg3ABC_config_.json"
basedir = os.getcwd()
rvg_dir = os.path.join(basedir, 'src', 'MCSimPython', 'vessels', 'gunnerus')

vesselABC = {}
vesselABC['Ar'] = Ar
vesselABC['Br'] = Br
vesselABC['Cr'] = Cr
vesselABC['MA'] = MA.tolist()
vesselABC['freqs'] = w.tolist()
