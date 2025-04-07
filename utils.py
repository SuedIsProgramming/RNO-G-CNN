from typing import Tuple, Dict, List
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import os


from torch.utils.data import Dataset
from typing import Tuple, Dict, List
from torch.nn.functional import normalize
import numpy as np
import torch

# Custom dataset class to work with dictionaries

class EventtoData(Dataset):
    """
    Custom Dataset class for handling event data.

    Attributes:
    -----------
    events : np.ndarray
        Array containing event data.
    n_channels : int
        Number of channels in the event data.
    n_bins : int
        Number of bins in the event data.
    bin_time : float
        Time duration of each bin.
    transform : callable, optional
        Optional transform to be applied on a sample (currently deprecated).

    Methods:
    --------
    __len__() -> int:
        Returns the number of events.
    __getitem__(index: int) -> Tuple[np.ndarray, float]:
        Returns a tuple (data, mean_SNR) for the given index.
    show_event(index: int):
        Displays the event data as an image using a utility function.
    mean_snr_of(index: int) -> float:
        Returns the mean SNR of the event at the given index.

    Notes:
    ------
    - The class now supports initialization with either an events dictionary or a file containing events.
    - The event data is converted from a dictionary to a numpy array for easier handling.
    - The `__getitem__` method ensures that the data includes a color dimension and converts data types to float32.
    """
    def __init__(self, events=None, events_f=None, normalize=None):
        if not events_f and not events:
            raise ValueError("Must include either events file or events to construct EventData object.")
        if events_f:
            with open(events_f, 'rb') as file:
                events_dict = pickle.load(file)
        else:
            events_dict = events

        # Must convert from dictionary to arrays
        self.events = np.array([events[key] for key in events.keys()])
        first_key = next(iter(events_dict))
        self.n_channels = events_dict[first_key]['data'].shape[0]
        self.n_bins = events_dict[first_key]['data'].shape[1]
        self.bin_time = events_dict[first_key]['bin_time'].item()

    def show_event(self, index : int):
        plot_image(self.events[index])

    def mean_snr_of(self, index : int):
        return self.events[index]['mean_SNR'].item()

    def __len__(self) -> int:
        return len(self.events)

    # Must overwrite __getitem__():
    def __getitem__(self, index : int) -> Tuple:
        item_data_uncolored = self.events[index]['data']
        item_data = torch.tensor(np.array([item_data_uncolored]).astype('float32')) # Must add a color dimension
        item_snr = torch.tensor(np.array(self.events[index]['mean_SNR']).astype('float32'))

        if normalize:
            item = (normalize(item_data),item_snr)
        else:
            item = (item_data,item_snr)

        return item # Returns (data, mean_SNR)

def make_predictions(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device = 'cpu', plot: bool = False, verbose: bool = False):
    """
    Make predictions using a trained PyTorch model on a given DataLoader.

    Args:
        model (torch.nn.Module): The trained PyTorch model to use for predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the data to predict on.
        device (torch.device, optional): The device to run the model on (default 'cpu').
        plot (bool, optional): Whether to plot the predicted vs true SNR values (default is False).
        verbose (bool, optional): Whether to print detailed prediction information, including MSE for each sample (default is False).

    Returns:
        list: A list of tuples containing predicted and true SNR values for each sample in the DataLoader.
    """

    predictions = []
    model.eval()
    with torch.inference_mode():
        for data, snr in dataloader:
            data = data.to(device)
            pred = model(data)
            predictions.append((pred.cpu().item(), snr.cpu().item()))

    if verbose:
        print("First 20 predictions (Predicted SNR, True SNR):")
        avg_mse = 0
        for i, (pred, true) in enumerate(predictions):
            print(f"Sample {i + 1}: Predicted = {pred:.16f}, True = {true:.16f}")
            mse = (pred-true)**2
            print(f'MSE: {mse}')
            avg_mse+=mse
        print(f'Average MSE: {avg_mse/20}')
            


    if plot:
        pred_values, true_values = zip(*predictions)
        samples = np.arange(len(pred_values))

        plt.figure(figsize=(12, 6))
        plt.plot(samples, pred_values, label='Predicted SNR', marker='o', linestyle='-', color='blue')
        plt.plot(samples, true_values, label='True SNR', marker='x', linestyle='--', color='red')
        plt.xlabel('Data Sample', fontsize=14)
        plt.ylabel('SNR', fontsize=14)
        plt.title('Predicted vs True SNR', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    return predictions

def find(name, path='/data/condor_shared/users/ssued/RNOGCnn'):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

    # Function to shift keys by a given number
    def shift_keys(d, shift_amount):
        return {k + shift_amount: v for k, v in d.items()}

def conjoin_events(input_path, file_path='/data/condor_shared/users/ssued/RNOGCnn/function_testing/data/eventbatch.pkl'):
    """
    Save events in a dictionary. If a dictionary already exists, it will append the events to the end of the dictionary.
    Otherwise, it will create the pickled dictionary in the file_path. Once the file has been merged, it will be deleted.

    Parameters:
    file_path (String) location of dictionary, or location where to create it.
    input_path (String) location of python dictionary from which to append the events or to save.
    """

    with open(input_path, 'rb') as file:
        events_in = pickle.load(file)
        print(f"Loaded events from {input_path}, number of events: {len(events_in)}")

    # Check if the file exists
    if os.path.exists(file_path): 
        print('Consolidating from:', input_path) 
        # Load and return the dictionary
        try:
            with open(file_path, 'rb') as file:
                event_dict = pickle.load(file)
                #print(f"Loaded existing event dictionary from {file_path}, number of events: {len(event_dict)}")
        except EOFError:
            print(f"Error: {file_path} is empty or corrupted.")
            event_dict = {}

        start_i = max(event_dict.keys(), default=-1)
        #print(f"Starting index for new events: {start_i + 1}")

        # Directly update the existing dictionary with shifted keys
        for k, v in events_in.items():
            event_dict[start_i + k + 1] = v

        with open(file_path, 'wb') as file:
            pickle.dump(event_dict, file)
            #print(f"Updated event dictionary saved to {file_path}, total number of events: {len(event_dict)}")
    else: # If file does not exist:
        #print('Creating new event dictionary')
        # Save the event dictionary to file 
        with open(file_path, 'wb') as file:
            pickle.dump(events_in, file)
            #print(f"New event dictionary saved to {file_path}, number of events: {len(events_in)}")

    os.remove(input_path) # Delete the file after merging
    #print(f"Deleted input file: {input_path}")

def save_events(directory='/data/condor_shared/users/ssued/RNOGCnn/function_testing/data/', events_in = None):
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

def get_script_path():
    abspath = os.path.abspath(__file__)
    script_dir = os.path.dirname(abspath)
    return Path(script_dir)

def obtain_evb(eventbatch : str, eventdata_folder : str = '/data/condor_shared/users/ssued/RNOGCnn/CNN_steps/eventdata'):
    """
    Quality of life function. Simply return event batch dictionary.

    Parameters:
    eventbatch (dict): Event dictionary to inspect.
    eventdata_folder (str): Folder where the eventbatch is stored.
    """
    import pickle
    if not eventbatch.endswith('.pkl'):
        eventbatch = f'{eventbatch}.pkl'
    with open(f'{eventdata_folder}/{eventbatch}', 'rb') as file:
        return pickle.load(file)
    
def plot_image(event):
    matrix = event['data']
    bintime_ns = event['bin_time']

    plt.figure(figsize=(10, 6))
    # Creating the heatmap
    plt.imshow(matrix, cmap='viridis', interpolation='nearest',aspect=5)
    plt.colorbar()  # adding color bar to show the scale
    plt.title("Heatmap of voltage on 4 channels")
    plt.xlabel(f"Bin Value:{bintime_ns: .3g} ns    ;    Mean SNR:{event['mean_SNR']: .3g}")
    y_ticks = [0, 1, 2, 3]  # Custom y-tick positions
    y_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']  # Custom y-tick labels
    plt.yticks(y_ticks, y_labels)
    plt.show()
