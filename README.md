# Wave-Model
Wave model simulator used in master project and master thesis


## Previous work
The simulator will be based on the work of ...... and Brørby 2021/2022. 

## Tips

### Setup virtual environment and install dependencies

Create a virtual environment e.g `py -m venv name-of-venv`.

Install required dependencies by running: `pip install -r requirements.txt`

### Relative Imports

When running different scripts you might encounter problems due to relative imports of packages/modules. This will typically be "No module named wave_spectra" etc. 
This has to do with how python looks for modules. 

In this directory, the modules wave_loads, wave_spectraw, and wave_sim are all in the base directory WAVEMODEL. To make sure that these modules can be found, we have to add this directory to the PYTHONPATH variable. You can check which directories python looks for packages by:

```
>>> import sys
>>> print(sys.path)
```

To ensure that the correct base directory `some\\path\\Wave-Model` can be found - we can add it to our virtual environment (e.g venv). Add a .pth file like `wavemodel.pth` in venv\Lib\site-packages containing the full path to the directory e.g:

```
wavemodel.pth
-------------

c:\Users\your_user\full_path\Wave-Model
```

The structure should look like:

```
some_folder
|___Wave-Model
|   |
|   |___Readme.md
|   |___demos/
|   |___simulator/
|   |___waves/
|   |___requirements.txt
|   |___ ...
|
|___name-of-venv
|   |
|   |___Lib
|       |
|       |___site-packages
|           |___wavemodel.pth
|           |___ ...
```
