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

os.chdir('/data/i3home/ssued/RNOGCnn/CNN_steps/data') # Changes working directory so that all steps occur in the "data" file.

param = NuRadioReco.framework.parameters.channelParameters # Parameter enumerator

event_reader = NuRadioReco.modules.io.eventReader.eventReader()

out_file = '/data/i3home/ssued/RNOGCnn/CNN_steps/data/output.nur' # Added absolute path, may have to change later
event_reader.begin(out_file)
events = event_reader.run()

# Unnecesary unless we want the total number of iterations.
# def count_iterable(i): # iterable function counts and returns number of iterables.
#     return sum(1 for e in i)

bin_n = 25 # States number of bins

event_dict = {} # Event dictionary will contain n dictionaries for each event with {'mean_SNR','data'} key pairs.

for iEvent, event in enumerate(events): # For each event
    print(f'Event: {iEvent}')

    for iStation, station in enumerate(event.get_stations()): # For each station
        print(f'> Station: {iStation}')
        v_matrix = np.empty((0, bin_n)) # initialize matrix that will hold [ch_n,bin_n] voltage data
        SNR_mean = 0 # enumerator
        bin_time = 0

        for iChannel, channel in enumerate(station.iter_channels()):
            print(f'>> Channel: {iChannel}')
            SNR_mean += channel.get_parameter(param.SNR)['peak_amplitude'] # Add SNRs for each channel (apparently peak_amplitude = SNR)
            v_hilb = channel.get_hilbert_envelope()
            t = channel.get_times()
            ch_data = np.array([t,v_hilb])
            _,binned_v,bin_time = utils.bin_v(ch_data,bin_n)
            v_matrix = np.vstack((v_matrix,binned_v)) # Stacks each binned voltage to form the matrix

        SNR_mean = SNR_mean / station.get_number_of_channels() # Divide by number of channels
        event_dict[iEvent] = {'mean_SNR' : SNR_mean, 'bin_time' : bin_time, 'data' : v_matrix} # Populate event dictionary

utils.save_events(events_in=event_dict,directory='/data/i3home/ssued/RNOGCnn/CNN_steps/eventdata') # Save the events