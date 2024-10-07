import NuRadioReco.modules.io.eventReader
from scipy.signal import hilbert # pylint: disable=W0622
import numpy as np
import pickle
import os

event_reader = NuRadioReco.modules.io.eventReader.eventReader()

out_file = 'output.nur'
event_reader.begin(out_file)
events = event_reader.run()

def save_events(file_path='data/event_data.pkl',events_in=None):
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
    else:
        # Save the event dictionary to file 
        with open(file_path, 'wb') as file:
            pickle.dump(events_in, file)

# Unnecesary unless we want the total number of iterations.
# def count_iterable(i): # iterable function counts and returns number of iterables.
#     return sum(1 for e in i)

# numevents = count_iterable(event_reader.run()) # Return num. of events

e_dict = {} # Will contain, 3d (channel,time,voltage) associated to each event.

for iE, event in enumerate(events):
    primary = event.get_primary() # Not sure what this is for D:
    print(f'Saving event {iE+1} info...')
    for iStation, station in enumerate(event.get_stations()): # For now, only one station.
        e_data = [] # Will contain channel num and voltage vs time trace.
        for ch in station.iter_channels():
            volts_hilb = abs(hilbert(ch.get_trace())) # Will save hilbert envelope.
            times = ch.get_times()
            e_data.append(np.array([times,volts_hilb])) # Appends voltage vs time trace to each channel.

        e_data = np.array(e_data) # Converts data from list to nparray.
        e_dict[iE] = e_data # Populates dictionary with event:3dArray pair.

save_events('/data/i3home/ssued/RNO-G-CNN/function_testing/data/event_dict.pkl',e_dict)

# Want to get data saved in following format:
# {batch_label, channels, width, height} would only have 1 channel