#!/data/i3home/ssued/bin/python

import NuRadioReco.modules.io.eventReader
from scipy.signal import hilbert # pylint: disable=W0622
import NuRadioReco.framework.parameters as parameters
import numpy as np
import pickle
import os
import sys
sys.path.append('/data/i3home/ssued/RNOGCnn') # Necessary to import utils from RNOGCnn directory
import utils

os.chdir('/data/i3home/ssued/RNOGCnn/CNN_steps/eventdata') # Changes working directory so that all steps occur in the "eventdata" file.

base_name = "eventbatch_"
ext = ".pkl"

for file_name in os.listdir(os.getcwd()): # Looks at all files in the directory
    if file_name.startswith(base_name) and file_name.endswith(ext): # If the file starts and ends with the given strings, run conjoin function on them.
        try:
            utils.conjoin_events(input_path=file_name, file_path='/data/i3home/ssued/RNOGCnn/CNN_steps/eventdata/eventbatch.pkl')
        except EOFError:
            print(f"Error: {file_name} is empty or corrupted.")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

print('Done conjoining events.')

# Debugging memory usage
import resource
usage=resource.getrusage(resource.RUSAGE_SELF)
memory_in_mb = usage[2]/1024.
print(f"Step 4 Mem usage {memory_in_mb} MB")