# MCSimPython

Vessel simulator used in master project and master thesis of M. Kongshaug, H. Mo and J. Hygen.

The simulator contains a set of simulation models in addition to wave loads.

The python package is not complete and there is no guarantee for the validity of the vessel models.

All code implementation is found in the `src` directory. The python package is structured as follows:

- `src/MCSimPython/simulator/` Simulation Models. *Python Path*:= `MCSimPython.simulator`
- `src/MCSimPython/waves/` Wave kinematics, wave spectra, and wave loads. *Python Path* := `MCSimPython.waves`
- `src/MCSimPython/guidance/` Reference Models. *Python path*:= `MCSimPython.guidance` *(in development)*
- `src/MCSimPython/observers/` Observers : `MCSimPython.observer` *(not implemented yet)*
- `src/MCSimPython/control` Controllers : `MCSimPython.control` *(not implemented yet)*

## Demos

Demonstration of how the individual components of the python package can be used is given in `demos`. Some demos of combination of the different components are also given here. 

