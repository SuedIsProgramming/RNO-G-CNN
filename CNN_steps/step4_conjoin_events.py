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

for file_name in os.listdir(os.getcwd()):
    if file_name.startswith(base_name) and file_name.endswith(ext):
        try:
            utils.conjoin_events(input_path=file_name, file_path='/data/i3home/ssued/RNOGCnn/CNN_steps/eventdata/eventbatch.pkl')
        except EOFError:
            print(f"Error: {file_name} is empty or corrupted.")
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

print('Done conjoining events.')