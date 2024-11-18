import NuRadioReco.modules.io.eventReader
from scipy.signal import hilbert # pylint: disable=W0622
import NuRadioReco.framework.parameters as parameters
import numpy as np
import pickle
import os

param = NuRadioReco.framework.parameters.channelParameters # Parameter enumerator

event_reader = NuRadioReco.modules.io.eventReader.eventReader()

out_file = '/data/i3home/ssued/RNOGCnn/CNN_steps/output.nur' # Added absolute path, may have to change later
event_reader.begin(out_file)
events = event_reader.run()

def save_events(file_path='/data/i3home/ssued/RNOGCnn/function_testing/data/event_dict.pkl',events_in=None):
    """
    Save events in a dictionary. If a dictionary already exists, it will append the events to the end of the dictionary.
    Otherwise, it will create the pickled dictionary in the file_path.

    Parameters:
    file_path (String) location of dictionary, or location where to create it.
    events_in (dict) python dictionary from which to append the events or to save.
    """
    # Check if the file exists
    if os.path.exists(file_path):   
        # Load and return the dictionary
        with open(file_path, 'rb') as file:
            event_dict = pickle.load(file)

        start_i = max(event_dict.keys())

        # Function to shift keys by a given number
        def shift_keys(d, shift_amount):
            return {k + shift_amount: v for k, v in d.items()}
        
        shifted_events_in = shift_keys(events_in,start_i+1)
        new_event_dict = {**event_dict, **shifted_events_in}

        with open(file_path, 'wb') as file:
            pickle.dump(new_event_dict, file)
    else: # If file does not exist:
        # Save the event dictionary to file 
        with open(file_path, 'wb') as file:
            pickle.dump(events_in, file)

# Unnecesary unless we want the total number of iterations.
# def count_iterable(i): # iterable function counts and returns number of iterables.
#     return sum(1 for e in i)

event_dict = {} # Event dictionary will contain n dictionaries for each event with {'mean_SNR','data'} key pairs.

for iEvent, event in enumerate(events): # For each event
    print(f'Event: {iEvent}')
    e_data = [] # initialize data of this event.

    for iStation, station in enumerate(event.get_stations()): # For each station
        print(f'> Station: {iStation}')
        intpow_mean = 0 # enumerator

        for iChannel, channel in enumerate(station.iter_channels()):
            print(f'>> Channel: {iChannel}')
            intpow_mean += channel.get_parameter(param.SNR)['peak_amplitude'] # Add SNRs for each channel (apparently peak_amplitude = SNR)
            v_hilb = channel.get_hilbert_envelope()
            t = channel.get_times()
            e_data.append(np.array([t,v_hilb])) # Appends voltage vs time trace of each channel

        intpow_mean = intpow_mean / station.get_number_of_channels() # Divide by number of channels
        event_dict[iEvent] = {'mean_SNR' : intpow_mean, 'data' : np.array(e_data)} # Populate event dictionary

save_events(events_in=event_dict) # Save the events

print('Done!')