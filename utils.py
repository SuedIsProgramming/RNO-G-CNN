import NuRadioReco.framework.parameters as parameters
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def find(name, path='/data/i3home/ssued/RNOGCnn'):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def conjoin_events(input_path,file_path='/data/i3home/ssued/RNOGCnn/function_testing/data/eventbatch.pkl'):
    """
    Save events in a dictionary. If a dictionary already exists, it will append the events to the end of the dictionary.
    Otherwise, it will create the pickled dictionary in the file_path. Once the file has been merged, it will be deleted.

    Parameters:
    file_path (String) location of dictionary, or location where to create it.
    input_path (String) location of python dictionary from which to append the events or to save.
    """

    with open(input_path, 'rb') as file:
        events_in = pickle.load(file)

    # Check if the file exists
    if os.path.exists(file_path): 
        print('Consolidating from:', input_path) 
        # Load and return the dictionary
        with open(file_path, 'rb') as file:
            event_dict = pickle.load(file)

        start_i = max(event_dict.keys())
        #print('Putting in this dictionary:', events_in.keys())
        #print('Into this dictionary:', event_dict.keys())

        # Function to shift keys by a given number
        def shift_keys(d, shift_amount):
            return {k + shift_amount: v for k, v in d.items()}
        
        shifted_events_in = shift_keys(events_in,start_i+1)
        new_event_dict = {**event_dict, **shifted_events_in}

        with open(file_path, 'wb') as file:
            pickle.dump(new_event_dict, file)
    else: # If file does not exist:
        print('File exists')
        # Save the event dictionary to file 
        with open(file_path, 'wb') as file:
            pickle.dump(events_in, file)

    os.remove(input_path) # Delete the file after merging

def save_events(directory='/data/i3home/ssued/RNOGCnn/function_testing/data/', events_in = None):
    """
    Save a file in the directory with a numerical suffix based on existing files.
    If no files exist, it saves as 'eventbatch_0.pkl'. Otherwise, it finds the highest numerical suffix,
    increments it, and saves the file.

    Parameters:
    directory (str): Directory to save the file.
    event_in (str): Event content to save in the file.
    """
    base_name = "eventbatch"
    ext = ".pkl"
    max_suffix = -1

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Iterate through the files in the directory to find the highest numerical suffix
    for file_name in os.listdir(directory):
        if file_name.startswith(base_name) and file_name.endswith(ext):
            try:
                # Extract the number after the base_name and before the extension
                suffix = int(file_name[len(base_name) + 1: -len(ext)])
                max_suffix = max(max_suffix, suffix)
            except ValueError:
                # Ignore files that don't follow the naming pattern
                pass

    # Determine the new file name
    new_suffix = max_suffix + 1
    new_file_name = f"{base_name}_{new_suffix}{ext}"
    new_file_path = os.path.join(directory, new_file_name)

    # Save the text content to the new file
    with open(new_file_path, 'wb') as file:
        pickle.dump(events_in, file)

    print(f"File saved as: {new_file_path}")

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