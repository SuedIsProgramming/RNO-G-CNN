from scipy.signal import hilbert
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Import dictionary. The dictionary is structured as follows: {Event1 : [channel1,channel2,channel3,channel4], ...}
# and each channel contains a 2D array of votlage (in 0th poisition) and time (in 1st position).
with open('saved_dictionary.pkl', 'rb') as f:
    event_dict = pickle.load(f)

def shift_relative_time(event, plot=False):
    """
    Shift all channels in an event by the value of the first trigger signal time.

    NOTE: Might be too expensive, as you have to search throughout all of the time arrays to find the minimum time value.
    Perhaps its better to not search? Looking at the generated voltage vs. time arrays, all channel seem to have the same starting time?

    Parameters:
    event (np.array): Event containing channels with voltage vs. time arrays.
    plot (bool): If True, generates before and after subplots (for debugging purposes).
    """
    
    # Plot before the shift, if requested
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(event[0][0], event[0][1])
        ax[0].set_title('Before Shift')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Voltage')

    # Find the minimum time across all channels
    min_time = min(np.min(channel[0]) for channel in event)

    # Subtract min_time from the time values of each channel
    for channel in event:
        channel[0] -= min_time

    # Plot after the shift, if requested
    if plot:
        ax[1].plot(event[0][0], event[0][1])
        ax[1].set_title('After Shift')
        ax[1].set_xlabel('Time (ns)')
        ax[1].set_ylabel('Voltage (V)')
        plt.tight_layout()
        plt.savefig('Before_and_after_shift.png')

def calculate_noise(voltage_arr, plot=False):
    """
    Calculate the RMS noise of a voltage array and optionally plot the voltage data.

    The noise is calculated by:
    1. Identifying the peak signal (maximum value) in the array.
    2. Defining a range around the peak, covering 25% of the array's length to the left and right of the peak.
    3. Extracting the noise subarrays from the portions of the array outside this range.
    4. Calculating the RMS noise from the concatenated noise subarrays.

    Parameters:
    voltage_arr (np.array): Array of voltage values.
    plot (bool): If True, plots the voltage data with noise range markers (For debugging purposes).

    Returns:
    float: The RMS noise value.
    """
    
    peak_idx = np.argmax(voltage_arr)
    quarter_len = len(voltage_arr) // 4

    # Define the range around the peak signal
    left_idx = max(0, peak_idx - quarter_len)
    right_idx = min(len(voltage_arr), peak_idx + quarter_len)

    # Extract noise subarrays and concatenate them
    noise_subarray = np.concatenate((voltage_arr[:left_idx], voltage_arr[right_idx:]))

    # Calculate RMS noise
    rms_noise = np.sqrt(np.mean(noise_subarray ** 2))

    # For testing purposes:
    if plot:
        # Plotting the original voltage data
        plt.figure(figsize=(10, 6))
        plt.plot(voltage_arr, label='Voltage Data', color='blue')
        
        # Mark the start and end indices of the noise range
        plt.axvline(x=left_idx, color='red', linestyle='--', label='Left Noise Index')
        plt.axvline(x=right_idx, color='green', linestyle='--', label='Right Noise Index')
        
        # Highlighting the noise areas on the plot
        plt.fill_between(range(left_idx), voltage_arr[:left_idx], color='red', alpha=0.3)
        plt.fill_between(range(right_idx, len(voltage_arr)), voltage_arr[right_idx:], color='green', alpha=0.3)
        
        plt.xlabel('Index')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage Data with Noise Range')
        plt.legend()
        plt.grid(True)
        plt.savefig('Voltage Data with Noise Range.png')

    return rms_noise

def bin_hilbert(channel, nbins, method='max', plot=False):
    """
    Bin Hilbert envelope vs. time data. The bin values are calculated according to different methods.

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

    # Obtain Hilbert envelope
    hilb_arr = np.abs(hilbert(voltage_arr))
    
    bin_width = len(voltage_arr) // nbins
    bin_time = time_arr[bin_width]-time_arr[0]
    binned_time = time_arr[:nbins * bin_width:bin_width]

    if method == 'max':
        binned_hilb = np.array([np.max(hilb_arr[i:i+bin_width]) for i in range(0, nbins * bin_width, bin_width)])
    elif method == 'avg':
        binned_hilb = np.array([np.mean(hilb_arr[i:i+bin_width]) for i in range(0, nbins * bin_width, bin_width)])
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'max' or 'avg'.")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_arr, hilb_arr, label='Original Hilbert Envelope', alpha=0.7)
        plt.plot(binned_time, binned_hilb, 'o-', label=f'Binned Hilbert Envelope ({method})', color='red')
        plt.xlabel(f'Time (ns), one bin = {bin_time} ns')
        plt.ylabel('Hilbert Envelope (V)')
        plt.title('Original vs Binned Hilbert Envelope')
        plt.legend()
        plt.grid(True)
        plt.savefig('Original vs Binned Hilbert Envelope.png')

    return binned_time, binned_hilb, bin_time

bin_hilbert(event_dict['Event3'][3],40,plot='True')

def plot_image(event, nbins):
    """
    Generates a heat map with the binned Hilbert envelope data of the signal. 
    The heat map has channel number on the vertical axis, time on the horizontal axis, 
    and voltage on the color axis.
    
    Parameters:
    event (np.array): Event containing channels with voltage vs. time arrays.
    nbins (int): Number of bins for the Hilbert envelope.
    """
    shift_relative_time(event)
    voltage_matrix = []
    time_bins = []
    
    for channel in event:
        _, voltage_bin, bin_time = bin_hilbert(channel, nbins, plot=False)
        voltage_matrix.append(voltage_bin)
        time_bins.append(bin_time)

    # Convert the list of voltage bins to an array
    voltage_matrix = np.array(voltage_matrix)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(voltage_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    # Set x and y axis labels
    plt.xlabel(f'dt: ({time_bins[0]} ns)')
    plt.ylabel('Channel')
    plt.colorbar(label='Hilbert Voltage (V)')

    # Set y-ticks to show channel numbers
    plt.yticks(ticks=np.arange(0, len(event)), labels=np.arange(0, len(event)))

    plt.savefig('heatmap_test.png')
    
# Most pressing issue:
# Identify whether or not events and channels can have different time domains. If not, then we are good, otherwise must find a work around as binning will be different for different channels.
# Identify optimal function structure. Perhaps, plot image should only plot images, not generate arrays?
# Begin testing!!