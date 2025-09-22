import os



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
        horses2plt_RELATIVE_DIR = "Solver/bin/horses2plt"
        horses2plt_ABS_DIR = os.path.join(SOLVER_PATH,horses2plt_RELATIVE_DIR)
        
        if not os.path.isfile(horses2plt_ABS_DIR): #check if horses2plt exists in the directory provided by user
            raise Exception("The horses2plt utility is not found.")

        # solution files:
        hsol_files = [file for file in os.listdir(SOL_PATH) if file.endswith(f".hsol")] #makes a list of hsol files
        
        if not hsol_files: 
            raise Exception("No .hsol found.")
        else:
            print(">>> Nº of .hsol files: ",len(hsol_files))
            
        # mesh files:
        mesh_files = [file for file in os.listdir(MESH_PATH) if file.endswith(f".hmesh")] #makes a list of hmesh files
        
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

