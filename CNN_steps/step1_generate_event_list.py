#!/data/i3home/ssued/venv_ubu22.04/bin/python3

from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

import argparse # Argument parser required to add a simulation number suffix to the file name

parser = argparse.ArgumentParser(description='Argument for file differentiation')
parser.add_argument('sim_num', type=str,
                    help='Number of simulation')

args = parser.parse_args()

sim_num = args.sim_num

# Obtain directory of this script
import os
from pathlib import Path
scriptd = os.path.dirname(os.path.abspath(__file__))
scriptd_path = Path(scriptd)

os.chdir(scriptd_path / 'symdata') # Change to symdata directory

# Setup logging
from NuRadioReco.utilities.logging import _setup_logger
logger = _setup_logger(name="")

# define simulation volume (artificially close by to make them trigger)
volume = {
'fiducial_zmin':-1 * units.km,  # the ice sheet at South Pole is 2.7km deep
'fiducial_zmax': 0 * units.km,
'fiducial_rmin': 0 * units.km,
'fiducial_rmax': 1 * units.km}

# generate one event list at 1e19 eV with 1000 neutrinos
generate_eventlist_cylinder(f'1e19_n1e3_{sim_num}.hdf5', 1e2, 1e19 * units.eV, 1e19 * units.eV, volume)

# Debugging memory usage
# import resource
# usage=resource.getrusage(resource.RUSAGE_SELF)
# memory_in_mb = usage[2]/1024.
# print(f" Step 1: Mem usage {memory_in_mb} MB")