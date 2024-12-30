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
        utils.conjoin_events(input_path=file_name)

print('Done conjoining events.')