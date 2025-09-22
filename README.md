# py_horses_dada_handler
Handle the solution data provided by HORSES3D CFD solver. 

## 1. Installation:

1. Make a local copy of the lib

    ```$ git clone https://github.com/pablo-alcaniz/py_horses_data_handler.git ```
2. Install the lib in the desired project with you env manager (i.e. uv/pip):

    ```$ uv add [lib installation directory]```
    or
    ```$ pip install -e [lib installation directory]```

## 2. Basic configuration:
The first step is to import the ```HorsesDataHandler``` class:

``` python
from py_horses_data_handler import HorsesDataHandler
```
And then just create an object calling the class: 

``` python
data_handler = HorsesDataHandler()
``` 

## 3. Features:

### 1. ```convert_hsol2hdf```
This function get the ```.hsol``` solution binaries and convert them to usable ```.hdf``` [(HDF5)](https://www.hdfgroup.org/solutions/hdf5/) files in a desired location. 

``` python
convert_hsol2hdf(SOLVER_PATH:str, SOL_PATH: str, MESH_PATH: str, HDF_PATH:str = "", output_parameters:str = "", log:bool = False) -> None
```
- Args:
    - ```SOLVER_PATH (str)```: Path of a HORSES3D valid installation (make sure to compile the solver with, at least, the HDF flag activated).
    - ```SOL_PATH (str)```: Path of the .hsol files.
    - ```MESH_PATH (str)```: Path of the .hmesh file.
    - ```HDF_PATH (str, optional)```: Path where you want your .hdf files at. Defaults to ""^[1].
    - ```output_parameters (str, optional)```: Flags for the horses2plt utility. Defaults to ""^[2].
    - ```log (bool, optional)```: Activates the stdout of horses2plt. Defaults to False.
- Returns:
    - ```None```

[1] The default directory of the ```.hdf``` is ```SOL_PATH```.

[2] The default ```output_parametres``` are ```"--output-mode=FE --output-variables=rho,u,v,w,p,T,Mach --output-type=vtkhdf"```.

Basic use:
```python
data_handler = HorsesDataHandler()

solver_path = "..."
sol_path = "..."
sol_mesh = "..."
hdf_path = "..."

data_handler.convert_hsol2hdf(solver_path, sol_path, sol_mesh, hdf_path)
```