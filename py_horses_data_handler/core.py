import os
import subprocess
import h5py
import re
from typing import Optional

def _get_time(stdout: Optional[str]) -> float:
    """Extract Time value from a horses2plt stdout string.

    Args:
        stdout (Optional[str]): The stdout string from horses2plt.

    Raises:
        ValueError: If stdout is empty or None.
        ValueError: If 'Time' is not found in stdout.
        ValueError: If 'Time' value cannot be parsed as float.

    Returns:
        float: The extracted Time value.
    """
    if not stdout:
        raise ValueError("stdout is empty or None; cannot extract Time")
    time_match = re.search(r'Time:\s*([\d\.]+)', stdout)
    if not time_match:
        # include a short slice of stdout to help debugging
        snippet = stdout[:200].replace("\n", "\\n")
        raise ValueError(f"Could not find 'Time' in output. stdout snippet: {snippet!r}")
    try:
        return float(time_match.group(1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to parse Time value: {time_match.group(1)!r}") from exc


def _get_iteration(stdout: Optional[str]) -> int:
    """
    Extract Iteration value from a horses2plt stdout string.

    Args:
        stdout (Optional[str]): The stdout string from horses2plt.

    Raises:
        ValueError: If stdout is empty or None.
        ValueError: If 'Iteration' is not found in stdout.
        ValueError: If 'Iteration' value cannot be parsed as int.

    Returns:
        int: The extracted Iteration value.
    """
    if not stdout:
        raise ValueError("stdout is empty or None; cannot extract Iteration")
    iter_match = re.search(r'Iteration:\s*(\d+)', stdout)
    if not iter_match:
        snippet = stdout[:200].replace("\n", "\\n")
        raise ValueError(f"Could not find 'Iteration' in output. stdout snippet: {snippet!r}")
    try:
        return int(iter_match.group(1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to parse Iteration value: {iter_match.group(1)!r}") from exc


class HorsesDataHandler:
    def __init__(self) -> None: 
        print(">>> Data handler started...")
    
    def convert_hsol2hdf(self, SOLVER_PATH:str, HSOL_DIR:str, MESH_DIR:str, OUT_FILES_DIR:Optional[str], 
                        OUTPUT_PARAMETERS: Optional[str], OUT_EXTENSION:str = ".h5", LOG:bool = False,
                        WRITE_SIM_PARAMETER:bool = True, EXTENDED_LOG:bool = True) -> None:
        """
        Function to convert .hsol binaries to .h5/.hdf with horses2ply utility.

        Args:
            SOLVER_PATH (str): Path of a HORSES3D valid installation (make sure to compile the solver with, at least, the HDF flag activated).
            HSOL_DIR (str): Path of the .hsol files.
            MESH_DIR (str): Path of the .hmesh file.
            OUT_FILES_DIR (str, optional): Path to save the output files. If None, uses HSOL_DIR. Defaults to None.
            OUTPUT_PARAMETERS (str, optional): Flags for the horses2plt utility. Defaults to "--output-mode=FE --output-variables=rho,u,v,w,p,T,Mach --output-type=vtkhdf".
            OUT_EXTENSION (str, optional): Extension of the output files. Defaults to "h5". Options are: ".h5", ".hdf" or "both".
            LOG (bool, optional): Activates the stdout of horses2plt. Defaults to False.
            WRITE_SIM_PARAMETER (bool, optional): If True, writes the simulation time and iteration in the .hdf files. Defaults to True. Please, note that this option slow down the conversion process around 4x times.
            EXTENDED_LOG (bool, optional): If True, shows the full stdout of horses2plt. Defaults to True.
        """
        
        ## Initial configurations:
        # Check and set default for OUT_FILES_DIR:
        if OUT_FILES_DIR is None:
            OUT_FILES_DIR = HSOL_DIR
        # Parameters config:            
        if OUTPUT_PARAMETERS is None:
            OUTPUT_PARAMETERS = "--output-mode=FE --output-variables=rho,u,v,w,p,T,Mach --output-type=vtkhdf"
        
        ## Initial checks:
        if not os.path.isdir(OUT_FILES_DIR):
            os.makedirs(OUT_FILES_DIR)   
        if OUT_EXTENSION not in [".h5", ".hdf", "both"]:
            raise Exception("The output extension must be either '.h5', '.hdf' or 'both'.")
        if not os.path.isdir(SOLVER_PATH):
            raise Exception("The solver path provided does not exist.")
        if not os.path.isdir(HSOL_DIR):
            raise Exception("The .hsol path provided does not exist.")
        if not os.path.isdir(MESH_DIR):
            raise Exception("The .hmesh path provided does not exist.")
        
        ## DIR checks and manipilation:
        # horses2plt:
        horses2plt_RELATIVE_DIR: str = "Solver/bin/horses2plt"
        horses2plt_ABS_DIR: str = os.path.join(SOLVER_PATH,horses2plt_RELATIVE_DIR)
        
        if not os.path.isfile(horses2plt_ABS_DIR): #check if horses2plt exists in the directory provided by user
            raise Exception("The horses2plt utility is not found.")

        # solution files:
        hsol_files: list[str] = [file for file in os.listdir(HSOL_DIR) if file.endswith(f".hsol")] #makes a list of hsol files
        
        if not hsol_files: 
            raise Exception("No .hsol found.")
        else:
            if LOG:
                print(">>> Nº of .hsol files: ",len(hsol_files))
            
        # mesh files:
        mesh_files: list[str] = [file for file in os.listdir(MESH_DIR) if file.endswith(f".hmesh")] #makes a list of hmesh files
        
        if not mesh_files: 
            raise Exception("No .hmesh found.")
        else:
            if LOG:
                print(">>> Nº of .hmesh files: ",len(mesh_files))
        
        # shell LOG config:
        if EXTENDED_LOG:
            LOG_control: str = ""
        else:
            LOG_control: str = "> /dev/null 2>&1"
    
        ## File generation:
        # Function to write time and iteration in the hdf file
        def write_data(file_path:str, time_value:float, iter_value:int) -> None:
            with h5py.File(file_path, "r+") as hdf_file:
                # access to vtkhdf group
                root_group = hdf_file['/VTKHDF'].require_group('SimulationInfo')
                # create or overwrite time attribute
                if 'Time' in root_group:
                    del root_group['Time']
                root_group.create_dataset('Time', data=time_value, dtype="float32")
                if 'Iteration' in root_group:
                    del root_group['Iteration']
                root_group.create_dataset('Iteration', iter_value, dtype="int32")
        
        # Convert each file and write time and iteration
        if WRITE_SIM_PARAMETER:
            for file in os.listdir(HSOL_DIR):
                if file.endswith('.hsol'):
                    base_name = os.path.splitext(file)[0]
                    convert_command = str(horses2plt_ABS_DIR+" "+os.path.join(HSOL_DIR,file)+" "+
                        os.path.join(MESH_DIR,mesh_files[0])+" "+OUTPUT_PARAMETERS)
                    process = subprocess.run(
                        convert_command,
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    # prefer stdout, but fall back to stderr for more information
                    command_output = process.stdout or process.stderr
                    time_value = _get_time(command_output)
                    iter_value = _get_iteration(command_output)
                    write_data(os.path.join(HSOL_DIR,base_name+".hdf"), time_value, iter_value)
                    if LOG:
                        print(f">>> Converted {file} at time {time_value}")
                        print(f">>> Time attribute written in {base_name+'.hdf'}")
                    if EXTENDED_LOG:
                        print(command_output)
        else:
            os.system(str(horses2plt_ABS_DIR+" "+os.path.join(HSOL_DIR,"*.hsol")+" "+
                          os.path.join(MESH_DIR,mesh_files[0])+" "+OUTPUT_PARAMETERS)+" "+LOG_control)

        def move_and_rename_files(hsol_path:str, out_path:str, extension:str) -> None:
            for file in os.listdir(hsol_path):
                if file.endswith('.hdf'):
                    base_name = os.path.splitext(file)[0]
                    os.rename(os.path.join(hsol_path,file), os.path.join(out_path,base_name+extension))
                    
        def copy_and_rename_files(hsol_path:str, out_path:str, extension:str) -> None:
            import shutil
            for file in os.listdir(hsol_path):
                if file.endswith('.hdf'):
                    base_name = os.path.splitext(file)[0]
                    shutil.copy2(os.path.join(hsol_path,file), os.path.join(out_path,base_name+extension))
        
        if OUT_EXTENSION == "both":
            if not os.path.isdir(os.path.join(OUT_FILES_DIR,"hdf")):
                os.makedirs(os.path.join(OUT_FILES_DIR,"hdf"))
            if not os.path.isdir(os.path.join(OUT_FILES_DIR,"h5")):
                os.makedirs(os.path.join(OUT_FILES_DIR,"h5"))
            copy_and_rename_files(HSOL_DIR, os.path.join(OUT_FILES_DIR,"hdf"), ".hdf")
            move_and_rename_files(HSOL_DIR, os.path.join(OUT_FILES_DIR,"h5"), ".h5")
        else:
            move_and_rename_files(HSOL_DIR, OUT_FILES_DIR, OUT_EXTENSION)

    def get_sizes(self, PATH: str, EXTENSION: str = ".hdf") -> None:
        """
        Get the size of all files (of a certain extension) in a certain directory

        Args:
            PATH (str): Path to examine.
            EXTENSION (str, optional): Extension of the files to search. Defaults to ".hdf".
        """
        
        files: list[str] = [file for file in os.listdir(PATH) if file.endswith(EXTENSION)]
        size: int = 0
        for _file_ in files:
            current_file: int = os.path.getsize(os.path.join(PATH,_file_))
            size = size + current_file
        if size < 1E9:
            print(">>> Data size:", float(size/1E6), "MB")
        else:
            print(">>> Data size:", float(size/1E9), "GB")
    
    def explore_hdf5_structure(self, FILEPATH: str) -> None:
        """
        Funcion to see the basic data configuration in a .hdf file

        Args:
            FILEPATH (str): Explicit path of the file to inspect
        """
        
        def print_structure(name, obj) -> None:
            indent = "     " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}: Group")
        
        print(f"File structure: {FILEPATH}")
        print("-" * 50)
        
        with h5py.File(FILEPATH, 'r') as f:
            f.visititems(print_structure)
        print("-" * 50)