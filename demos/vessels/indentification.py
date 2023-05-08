from MCSimPython.simulator import RVG_DP_6DOF
from MCSimPython.utils import system_identification as sys_id

import json
import numpy as np
import matplotlib.pyplot as plt

vessel = RVG_DP_6DOF(0.1, config_file="rvg2_config.json")
with open(vessel._config_file, 'r') as f:
    config = json.load(f)

A = np.asarray(config['A'])[:, :, :, 0]
B = np.asarray(config['B'])[:, :, :, 0]
w = np.asarray(config['freqs'])

MA = sys_id(w, A, B, max_order=9, method=0, plot_estimate=True)