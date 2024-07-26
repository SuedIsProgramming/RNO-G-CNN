import NuRadioReco.modules.io.eventReader
event_reader = NuRadioReco.modules.io.eventReader.eventReader()

file = 'output.nur'
event_reader.begin(file)
events = event_reader.run()

def count_iterable(i): # iterable function counts and returns number of iterations
    return sum(1 for e in i)

numevents = count_iterable(event_reader.run()) # Return num. of events

for iE, event in enumerate(events):
    #primary = event.get_primary() Not sure what this is for (?)
    print('Event number:', iE, '\n')
    for iStation, station in enumerate(event.get_stations()): # For beginning, only one station.
        print('Station number:', iStation)
        # this loops through "mock data" (with noise added, etc.)
        for ch in station.iter_channels():
            print('Channel number:', ch.get_trace().size)
            
            volts = ch.get_trace()
            times = ch.get_times()
            data = [ch.get_trace(),ch.get_times()]


# Issue with saving data. How should I save the traces? Will one data file have 1 event, or all events? Will I have to alter this for different stations?
# Because right now if I save one file per event and per station its good. But in the future, if I increment the number of stations, perhaps saving multiple stations is the way. But how do I separate it?
# Maybe a 4D array [stations,channels,time,voltage]?



# from matplotlib import pyplot as plt

# import NuRadioReco.modules.io.eventReader
# event_reader = NuRadioReco.modules.io.eventReader.eventReader()

# file = 'output.nur'
# event_reader.begin(file)
# for iE, event in enumerate(event_reader.run()):
#     primary = event.get_primary()

#     for iStation, station in enumerate(event.get_stations()):

#         # a fig and axes for our waveforms
#         fig, axs = plt.subplots(4, 1, figsize=(5,20))

#         # this loops through "mock data" (with noise added, etc.)
#         for ch in station.iter_channels():
#             volts = ch.get_trace()
#             times = ch.get_times()
#             axs[ch.get_id()].plot(times, volts)
#             axs[ch.get_id()].set_title(f"Channel {ch.get_id()}")
        
#         # this loops through *MC truth* waveforms (before noise was added)
#         # this may prove useful at some point
#         # if station.has_sim_station():
#         #     sim_station = station.get_sim_station()
#         #     for sim_ch in sim_station.iter_channels():
#         #         volts = sim_ch.get_trace()
#         #         times = sim_ch.get_times()
#         #         axs[sim_ch.get_id()].plot(times, volts, '--')

#         for ax in axs:
#             ax.set_xlabel("Time [ns]")
#             ax.set_ylabel("Voltage [V]")

#         fig.savefig(f"traces_{iE}.png") # save the traces
