import os
import h5py



class HorsesDataHandler():
    
    def __init__(self) -> None: 
        print(">>> Data handler started...")
    
    
    def convert_hsol2hdf(self, SOLVER_PATH:str, SOL_PATH: str, MESH_PATH: str, HDF_PATH:str = "", output_parameters:str = "", log:bool = False) -> None:
        """
        Function to convert .hsol binaries to .hdf with horses2ply utility.

        Args:
            SOLVER_PATH (str): Path of a HORSES3D valid installation (make sure to compile the solver with, at least, the HDF flag activated).
            SOL_PATH (str): Path of the .hsol files.
            MESH_PATH (str): Path of the .hmesh file.
            HDF_PATH (str, optional): Path where you want your .hdf files at. Defaults to "".
            output_parameters (str, optional): Flags for the horses2plt utility. Defaults to "".
            log (bool, optional): Activates the stdout of horses2plt. Defaults to False.
        """
        
        ## DIR checks and manipilation:
        # horses2plt:
        horses2plt_RELATIVE_DIR: str = "Solver/bin/horses2plt"
        horses2plt_ABS_DIR: str = os.path.join(SOLVER_PATH,horses2plt_RELATIVE_DIR)
        
        if not os.path.isfile(horses2plt_ABS_DIR): #check if horses2plt exists in the directory provided by user
            raise Exception("The horses2plt utility is not found.")

        # solution files:
        hsol_files: list[str] = [file for file in os.listdir(SOL_PATH) if file.endswith(f".hsol")] #makes a list of hsol files
        
        if not hsol_files: 
            raise Exception("No .hsol found.")
        else:
            print(">>> Nº of .hsol files: ",len(hsol_files))
            
        # mesh files:
        mesh_files: list[str] = [file for file in os.listdir(MESH_PATH) if file.endswith(f".hmesh")] #makes a list of hmesh files
        
        if not mesh_files: 
            raise Exception("No .hmesh found.")
        else:
            print(">>> Nº of .hmesh files: ",len(mesh_files))
            
        # Save dir for hdf file:
        if HDF_PATH == "": 
            HDF_PATH = SOL_PATH
        
        # Parameters config:
        if output_parameters == "":
            output_parameters = "--output-mode=FE --output-variables=rho,u,v,w,p,T,Mach --output-type=vtkhdf"
        
        # shell log config:
        if log:
            log_control = ""
        else:
            log_control = "> /dev/null 2>&1"
            
        
        ## File generation:
        os.system(str(horses2plt_ABS_DIR+" "+os.path.join(SOL_PATH,"*.hsol")+" "+
                      os.path.join(MESH_PATH,mesh_files[0])+" "+output_parameters)+" "+log_control)
        
        # Move files to the desired location 
        os.system(f"mv {SOL_PATH}/*.hdf {HDF_PATH}/")
        
        # Change the extension of the files to .h5
        for file in os.listdir(HDF_PATH):
            if file.endswith(".hdf"):
                base = os.path.splitext(file)[0]
                os.rename(os.path.join(HDF_PATH,file), os.path.join(HDF_PATH,base + ".h5"))

        # Size functionality
        hdf_files: list[str] = [file for file in os.listdir(HDF_PATH) if file.endswith(f".h5")]
        size: int = 0
        for file in hdf_files:
            current_file: int = os.path.getsize(os.path.join(HDF_PATH,file))
            size = size + current_file
        if size < 1000:
            print(">>> Data size:", float(size/1E6), "MB")
        else:
            print(">>> Data size:", float(size/1E9), "GB")
        

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
    
    def explore_hdf5_structure(self, filepath: str) -> None:
        """
        Funcion to see the basic data configuration in a .hdf file

        Args:
            filepath (str): Explicit path of the file to inspect
        """
        
        def print_structure(name, obj) -> None:
            indent = "     " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}: Group")
        
        print(f"File structure: {filepath}")
        print("-" * 50)
        
        with h5py.File(filepath, 'r') as f:
            f.visititems(print_structure)
        print("-" * 50)