from scipy.signal import hilbert
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

with open('data/saved_dictionary.pkl', 'rb') as file:
    event_dict = pickle.load(file)

def calculate_SNR(matrix, plot = False):
    SNR = 0
    for i,channel in enumerate(matrix):
        SNR += np.max(channel)/calculate_noise(channel, plot)

    SNR_mean = SNR/4

    return SNR_mean

def save_events(file_path='data/event_data.pkl',events_in=None):
    """
    Save events in a dictionary. If a dictionary already exists, it will append the events to the end of the dictionary.
    Otherwise, it will create the pickled dictionary in the file_path.

    Parameters:
    file_path (String) location of dictionary, or location where to create it.
    events_in (dict) python dictionary from which to append the events or to save.
    """

    event_data_to_append = {}

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

def bin_matrix(event,bins = 25, plotting = False):
    row1 = bin_hilbert(event[0],bins)[1]
    row2 = bin_hilbert(event[1],bins)[1]
    row3 = bin_hilbert(event[2],bins)[1]
    row4 = bin_hilbert(event[3],bins)[1]
    bintime = row1[3]
    bintime_ns = bintime*(10**6)
    matrix = np.array([row1,row2,row3,row4])

    if plotting:
        plt.figure(figsize=(10, 6))
        # Creating the heatmap
        plt.imshow(matrix, cmap='viridis', interpolation='nearest',aspect=5)
        plt.colorbar()  # adding color bar to show the scale
        plt.title("Heatmap of voltage on 4 channels")
        plt.xlabel(f"Bin Value:{bintime_ns: .3g} ns")
        y_ticks = [0, 1, 2, 3]  # Custom y-tick positions
        y_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']  # Custom y-tick labels
        plt.yticks(y_ticks, y_labels)
        plt.show()

    return matrix

#event_dict[]