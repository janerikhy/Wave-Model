# MCSimPython

Vessel simulator used in master project and master thesis of M. Kongshaug, H. Mo and J. Hygen.

The simulator contains a set of simulation models in addition to wave loads.

The python package is not complete and there is no guarantee for the validity of the vessel models.

All code implementation is found in the `src` directory. The python package is structured as follows:

- `src/MCSimPython/simulator/` Simulation Models. *Python Path*:= `MCSimPython.simulator`
- `src/MCSimPython/waves/` Wave kinematics, wave spectra, and wave loads. *Python Path* := `MCSimPython.waves`
- `src/MCSimPython/guidance/` Reference Models. *Python path*:= `MCSimPython.guidance` *(in development)*
- `src/MCSimPython/observers/` Observers : `MCSimPython.observer` *(in development)*
- `src/MCSimPython/control` Controllers : `MCSimPython.control` *(in development)*

## How to use MCSimPython

- Clone the GitHub repository to your local computer. 
- Create a virtual environment `py -m venv name-of-venv`
- Activate virtual environment `name-of-venv\scripts\activate`
- Update pip and setuptools: `py -m pip install --upgrade pip setuptools`
- Install the python package locally as an editable: `pip install -e .`
- Verify that the python package `MCSimPython` has been installed properly by running a demo script, or simply
```
(venv) C:\path\to\dir> python
>>> import MCSimPython
>>>
```


## Demos

Demonstration of how the individual components of the python package can be used is given in `demos`. Some demos of combination of the different components are also given here. 

