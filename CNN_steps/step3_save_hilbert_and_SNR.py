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

def bin_v(channel, nbins, hilbert = False, method='max', plot=False):
    """
    Bin voltage vs. time data. The bin values are calculated according to different methods.

    A big issue is that if the different channels effectively HAVE different time domains, then the different channels will contain
    different bin sizes. We could make all bins the same size, but then graphs will have more or less bins, or we could keep it like this.
    Parameters:
    channel (np.array): Channel array containing two numpy arrays of time (pos [0]) and voltage (pos [1]).
    method (str): Modifies bin values according to the method. 
                  'max' - Bins Hilbert envelope according to the local maximum of their respective time bins.
                  'avg' - Bins Hilbert envelope according to the average of their respective time bins.
    plot (bool): If True, plots the original Hilbert envelope and the binned data.

    Returns:
    np.array: Binned time array
    np.array: Binned Hilbert envelope array
    float: Time of a bin.
    """
    
    time_arr = channel[0]
    voltage_arr = channel[1]

    if hilbert is True:
        # Obtain Hilbert envelope
        v_arr = np.abs(hilbert(voltage_arr))
    else:
        v_arr = voltage_arr

    bin_width = len(voltage_arr) // nbins
    bin_dt = time_arr[bin_width]-time_arr[0]
    binned_time = time_arr[:nbins * bin_width:bin_width]

    if method == 'max':
        binned = np.array([np.max(v_arr[i:i+bin_width]) for i in range(0, nbins * bin_width, bin_width)])
    elif method == 'avg':
        binned = np.array([np.mean(v_arr[i:i+bin_width]) for i in range(0, nbins * bin_width, bin_width)])
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'max' or 'avg'.")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_arr, v_arr, label='Original Hilbert Envelope', alpha=0.7)
        plt.plot(binned_time, binned, 'o-', label=f'Binned Hilbert Envelope ({method})', color='red')
        plt.xlabel(f'Time (ns), one bin = {bin_dt} ns')
        plt.ylabel('Hilbert Envelope (V)')
        plt.title('Original vs Binned Hilbert Envelope')
        plt.legend()
        plt.grid(True)
        plt.savefig('Original vs Binned Hilbert Envelope.png')

    return binned_time, binned, bin_dt

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
            _,binned_v,bin_time = bin_v(ch_data,bin_n)
            v_matrix = np.vstack((v_matrix,binned_v)) # Stacks each binned voltage to form the matrix

        SNR_mean = SNR_mean / station.get_number_of_channels() # Divide by number of channels
        event_dict[iEvent] = {'mean_SNR' : SNR_mean, 'bin_time' : bin_time, 'data' : v_matrix} # Populate event dictionary

save_events(events_in=event_dict) # Save the events

print('Done! Data saved in: /data/i3home/ssued/RNOGCnn/function_testing/data/event_dict.pkl')