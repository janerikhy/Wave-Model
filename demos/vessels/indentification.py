from MCSimPython.simulator import RVG_DP_6DOF
from MCSimPython.utils import system_identification as sys_id

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Get desired file name
file_name = input("Enter file name: ")
if not file_name.endswith('.json'):
    file_name += '.json'
vessel_name = input("Enter vessel name (gunnerus / CSAD): ")
if vessel_name not in ['gunnerus', 'CSAD']:
    raise ValueError("Vessel name must be either 'gunnerus' or 'CSAD'.")
basedir = os.getcwd()
rvg_dir = os.path.join(basedir, 'src', 'MCSimPython', 'vessel_data', vessel_name)

vessel = RVG_DP_6DOF(0.1, config_file="rvg3_config.json")
with open(vessel._config_file, 'r') as f:
    config = json.load(f)

A = np.asarray(config['A'])[:, :, :, 0]
B = np.asarray(config['B'])[:, :, :, 0]
w = np.asarray(config['freqs'])

MA, Ar, Br, Cr = sys_id(w, A, B, max_order=6, method=2, plot_estimate=True)
with np.printoptions(precision=3, suppress=True):
    print(MA)
    
    
# Save the data to a configuration file to use it for simulation.

vesselABC = {}
vesselABC['Ar'] = sum(Ar, [])   # Flatten the list of lists (as in MATLAB)
vesselABC['Br'] = sum(Br, [])
vesselABC['Cr'] = sum(Cr, [])
vesselABC['MA'] = MA.tolist()
vesselABC['MRB'] = np.asarray(config['MRB']).tolist()
vesselABC['G'] = np.asarray(config['C'])[:, :, 0, 0].tolist()

vesselABC['freqs'] = w.tolist()

with open(os.path.join(rvg_dir, file_name), 'w') as f:
    json.dump(vesselABC, f)
