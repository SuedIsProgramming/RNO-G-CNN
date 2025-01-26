#!/data/i3home/ssued/venv_ubu22.04/bin/python3

from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation

# Obtain directory of this script
import os
from pathlib import Path
scriptd = os.path.dirname(os.path.abspath(__file__))
scriptd_path = Path(scriptd)

os.chdir(scriptd_path / 'symdata') # Change to symdata directory

# Setup logging
from NuRadioReco.utilities.logging import _setup_logger
logger = _setup_logger(name="")

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()


class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[80 * units.MHz, 1000 * units.GHz],
                                  filter_type='butter', order=2)
        channelBandPassFilter.run(evt, station, det, passband=[0, 500 * units.MHz],
                                  filter_type='butter', order=10)

    def _detector_simulation_trigger(self, evt, station, det):
        # first run a simple threshold trigger
        simpleThreshold.run(evt, station, det,
                             threshold=3 * self._Vrms,
                             triggered_channels=None,  # run trigger on all channels
                             number_concidences=1,
                             trigger_name='simple_threshold')  # the name of the trigger

        # run a high/low trigger on the 4 downward pointing LPDAs
        highLowThreshold.run(evt, station, det,
                                    threshold_high=4 * self._Vrms,
                                    threshold_low=-4 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_4.1sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
parser.add_argument('sim_num', type=str,
                    help='Number of simulation')
args = parser.parse_args()


if __name__ == "__main__":
    max_retries = 5
    retry_delay = 1  # initial delay in seconds

    for attempt in range(max_retries):
        try:
            sim = mySimulation(inputfilename=args.inputfilename,
                               outputfilename=args.outputfilename,
                               detectorfile=args.detectordescription,
                               outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                               config_file=args.config,
                               file_overwrite=True)
            sim.run()
            break  # exit the loop if the simulation runs successfully
        except BlockingIOError as e:
            logger.error(f"BlockingIOError occurred: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries})")
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # exponential backoff
            else:
                logger.error("Max retries reached. Exiting.")
                raise


# Debugging memory usage
# import resource
# usage=resource.getrusage(resource.RUSAGE_SELF)
# memory_in_mb = usage[2]/1024.
# print(f"Step 2 Mem usage {memory_in_mb} MB")