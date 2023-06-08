# MCSimPython

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python: v3.8](https://shields.io/badge/python-v3.8-green.svg)](https://www.python.org/downloads/release/python-380/) [![Python: v3.9](https://shields.io/badge/python-v3.9-green.svg)](https://www.python.org/downloads/release/python-390)

Vessel simulator used in master project and master thesis of M. Kongshaug, H. Mo and J. Hygen. The project is developed at Norwegian University of Science and Technology, Institue of Marine Technology. 

The python package is not complete and there is no guarantee for the validity of the vessel models.

The complete documentation can be found at: https://wave-model.readthedocs.io/en/latest/index.html

## How to use MCSimPython

### PyPi
The `MCSimPython` package can be installed from PyPi using `pip`:

`pip install MCSimPython`

### From GitHub:
Install from GitHub in the following:
- Clone the GitHub repository to your local computer.
- Create a virtual environment `py -m venv name-of-venv`
- Activate virtual environment `name-of-venv\scripts\activate`
- Update pip and setuptools: `py -m pip install --upgrade pip setuptools`
- Install the python package locally: `pip install .` (alternatively you can install as an editable `pip install -e .`)
- Verify that the python package `MCSimPython` has been installed properly by running a demo script, or simply in the command promt:
```
(venv) C:\path\to\dir> python
>>> import MCSimPython
>>>
```

## Demos

Demonstration of how the individual components of the python package can be used is given in `demos`. Some demos of combination of the different components are also given here. 

## Visualization

### RVG 6DOF model in beam sea
![6DOF RVG 2D Visualization](https://github.com/janerikhy/Wave-Model/blob/main/demos/animations/vessel_motion3d__rvg_waveangle_90.gif)

### RVG 6DOF model in head sea
![6DOF RVG 3D Visualization](https://github.com/janerikhy/Wave-Model/blob/main/demos/animations/vessel_motion3d__rvg_waveangle_180.gif)

<!---
### CSAD 6DOF model in multidirectional sea
![6DOF CSAD 2D Visualisation](https://github.com/janerikhy/Wave-Model/blob/main/demos/animations/vessel_motion3d_22.gif)
--->
