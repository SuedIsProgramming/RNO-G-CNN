#!/data/i3home/ssued/venv_ubu22.04/bin/python3

import NuRadioReco.modules.io.eventReader
from scipy.signal import hilbert # pylint: disable=W0622
import NuRadioReco.framework.parameters as parameters
import numpy as np
import pickle
import os
import sys
sys.path.append('/data/condor_shared/users/ssued/RNOGCnn') # Necessary to import utils from RNOGCnn directory
import utils

# Obtain directory of this script
from pathlib import Path
scriptd = os.path.dirname(os.path.abspath(__file__))
scriptd_path = Path(scriptd)

os.chdir(scriptd_path / 'eventdata') # Change to eventdata directory

base_name = "eventbatch_"
ext = ".pkl"

for file_name in os.listdir(os.getcwd()): # Looks at all files in the directory
    if file_name.startswith(base_name) and file_name.endswith(ext): # If the file starts and ends with the given strings, run conjoin function on them.
        try:
            utils.conjoin_events(input_path=file_name, file_path='/data/condor_shared/users/ssued/RNOGCnn/CNN_steps/eventdata/eventbatch.pkl')
        except EOFError:
            print(f"Error: {file_name} is empty or corrupted.")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

for file_name in os.listdir('/data/condor_shared/users/ssued/RNOGCnn/CNN_steps/symdata'): # Get rid of nur and hdf5 files that are no longer needed.
    if file_name.startswith('1e19_n1e3_') and file_name.endswith('.hdf5'):
        try:
            os.remove(f'/data/condor_shared/users/ssued/RNOGCnn/CNN_steps/symdata/{file_name}')
        except EOFError:
            print(f"Error: {file_name} is empty or corrupted.")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")
    elif file_name.startswith('output_') and file_name.endswith('.nur'):
        try:
            os.remove(f'/data/condor_shared/users/ssued/RNOGCnn/CNN_steps/symdata/{file_name}')
        except EOFError:
            print(f"Error: {file_name} is empty or corrupted.")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")
    elif file_name.startswith('output_') and file_name.endswith('.hdf5'):
        try:
            os.remove(f'/data/condor_shared/users/ssued/RNOGCnn/CNN_steps/symdata/{file_name}')
        except EOFError:
            print(f"Error: {file_name} is empty or corrupted.")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")


print('Done conjoining events.')

# Debugging memory usage
# import resource
# usage=resource.getrusage(resource.RUSAGE_SELF)
# memory_in_mb = usage[2]/1024.
# print(f"Step 4 Mem usage {memory_in_mb} MB")