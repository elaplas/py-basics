
import subprocess


###### running command lines in linux shell #############


def run_app(cmd:str, log_file:str):
    """Runs cmd
       cmd: command line e.g. a path of executable and its arguments
       log_file: path of a txt file for saving the logs of the executable e.g. debugging info logged out 
    """

    rect = subprocess.call(cmd, shell=True, stdout=log_file)
    if not rect:
        print(f"{cmd}: processed successfully! For more details see: {log_file}")
    else:
        print(f"{cmd}: failed! For more details see: {log_file}")
