import NuRadioReco.modules.io.eventReader
from scipy.signal import hilbert # pylint: disable=W0622
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

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

def bin_matrix(event,bins = 25, plotting = False, hilb = False):

    avg_int_pow = event['average_integrated_power']
    binned_dict = {'average_integrated_power' : avg_int_pow}

    row1 = bin_v(event['data'][0],bins, hilbert = hilb)[1]
    row2 = bin_v(event['data'][1],bins, hilbert = hilb)[1]
    row3 = bin_v(event['data'][2],bins, hilbert = hilb)[1]
    row4 = bin_v(event['data'][3],bins, hilbert = hilb)[1]
    bintime = row1[3]
    bintime_ns = bintime*(10**6)
    matrix = np.array([row1,row2,row3,row4])

    binned_dict = {'data' : matrix}

    if plotting:
        plt.figure(figsize=(10, 6))
        # Creating the heatmap
        plt.imshow(matrix, cmap='viridis', interpolation='nearest',aspect=5)
        plt.colorbar()  # adding color bar to show the scale
        plt.title("Heatmap of voltage on 4 channels")
        plt.xlabel(f"Bin Value:{bintime_ns: .3g} ns    ;    Mean Integrated Power:{event['mean_integrated_power']: .3g}")
        y_ticks = [0, 1, 2, 3]  # Custom y-tick positions
        y_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']  # Custom y-tick labels
        plt.yticks(y_ticks, y_labels)
        plt.show()

    return binned_dict

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

nbins = 25

with open('/data/i3home/ssued/RNOGCnn/function_testing/data/event_dict.pkl', 'rb') as file:
    event_dict = pickle.load(file)

binned_dict = {}
for iEvent , event in enumerate(event_dict):
    print(event)
    binned_event = bin_matrix(event,bins=nbins)
    binned_dict[iEvent] = binned_event

save_events(file_path='/data/i3home/ssued/RNOGCnn/function_testing/data/binned_event_dict.pkl',events_in=binned_dict)