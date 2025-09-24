# py_horses_dada_handler
Handle the solution data provided by HORSES3D CFD solver. 

## 1. Installation:

1. Make a local copy of the lib:

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
This function get the ```.hsol``` solution binaries and convert them to usable ```.h5``` [(HDF5)](https://www.hdfgroup.org/solutions/hdf5/)  files in a desired location. This function use the ```horses2plt``` build-in utility to make ```.hdf``` files, then, the files are converted to ```.h5``` extension for compatibility purposes. 

NOTE: for a correct visualization in Paraview ```.hdf``` extension are required.

``` python
convert_hsol2hdf(SOLVER_PATH:str, HSOL_DIR:str, MESH_DIR:str, OUT_FILES_DIR:Optional[str], OUTPUT_PARAMETERS: Optional[str], OUT_EXTENSION:str = ".h5", LOG:bool = False, WRITE_SIM_PARAMETER:bool = True, EXTENDED_LOG:bool = True) -> None:
```
- Args:
    - ```SOLVER_PATH (str)```: Path of a HORSES3D valid installation (make sure to compile the solver with, at least, the HDF flag activated).
    - ```HSOL_DIR (str)```: Directory of the .hsol files.
    - ```MESH_DIR (str)```: Directory of the .hmesh file.
    - ```OUT_FILES_DIR (str, optional)```: Path to save the output files. If None, uses ```HSOL_DIR```^[1]. Defaults to None.
    - ```OUTPUT_PARAMETERS (str, optional)```: Flags for the horses2plt utility. Defaults to "--output-mode=FE --output-variables=rho,u,v,w,p,T,Mach --output-type=vtkhdf".
    - ```OUT_EXTENSION (str, optional)```: Extension of the output files. Defaults to "h5". Options are: ".h5", ".hdf" or "both".
    - ```LOG (bool, optional)```: Activates the stdout of horses2plt. Defaults to False.
    - ```WRITE_SIM_PARAMETERS (bool, optional)```: If True, writes the simulation time and iteration in the .hdf files. Defaults to True. Please, note that this option slow down the conversion process around 4x times.
    - ```EXTENDED_LOG (bool, optional)```: If True, shows the full stdout of horses2plt. Defaults to True.
- Returns:
    - ```None```


Basic use:
```python
data_handler = HorsesDataHandler()

solver_path = "path/to/horses3d/installation"
hsol_path = "path/to/hsol/files"
sol_mesh = "path/to/mesh/file"
out_path = "path/to/output/files"

data_handler.convert_hsol2hdf(solver_path, hsol_path, sol_mesh, out_path)
```

### 2. ```get_sizes```

Get the size of all files (of a certain extension) in a certain directory.

```python
get_sizes(PATH: str, EXTENSION: str = ".hdf") -> None
```
- Args:
    - ```PATH (str)```: Path to examine.
    - ```EXTENSION (str, optional)```: Extension of the files to search. Defaults to ".hdf".
- Returns:
    - ```None```

### 3. ```explore_hdf5_structure```
Funcion to see the basic data configuration in a .hdf file.

```python
explore_hdf5_structure(FILEPATH: str) -> None:
```
- Args:
    - ```FILEPATH (str)```: Explicit path of the file to inspect.
- Returns:
    - ```None```

