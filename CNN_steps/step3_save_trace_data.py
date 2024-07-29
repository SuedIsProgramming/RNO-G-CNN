import NuRadioReco.modules.io.eventReader
import numpy as np

event_reader = NuRadioReco.modules.io.eventReader.eventReader()

file = 'output.nur'
event_reader.begin(file)
events = event_reader.run()

""" Unnecesary unless we want the total number of iterations.
def count_iterable(i): # iterable function counts and returns number of iterables.
    return sum(1 for e in i)

numevents = count_iterable(event_reader.run()) # Return num. of events
"""

e_dict = {} # Will contain, 3d (channel,time,voltage) associated to each event.

for iE, event in enumerate(events):
    primary = event.get_primary() # Not sure what this is for D:
    for iStation, station in enumerate(event.get_stations()): # For now, only one station.
        data = [] # Will contain channel num and voltage vs  time trace.
        for ch in station.iter_channels():
            volts = ch.get_trace()
            times = ch.get_times()
            data.append(np.array([times,volts])) # Appends voltage vs time trace to each channel.

        data = np.array(data) # Converts data from list to nparray.
        e_dict[f"Event{iE+1}"] = data # Populates dictionary with event:3dArray pair.