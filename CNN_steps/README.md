# Plotting a simulated event (eventually for machine learning)

## Pre-reqs
You will need a fully working installation of 
[NuRadioMC](https://github.com/nu-radio/NuRadioMC).
The docs for NRMC have installation instructions included,
but you can also follow Brian's guide [here](https://docs.google.com/document/d/1_DiXflPqdAiTX8RUd8hOcT9PNDT8WvN_MQrRbmat4jM/edit?usp=sharing).


## Running the sims

This has three steps.

### Step 1
Generate the events to be simulated. Run this like 
```
python step1_generate_event_list.py
```
This will produce an output file named `1e10_n1e3.hdf5` which will contain
1000 neutrinos with energies of 1E19 eV to be simulated.

### Step 2
Actually run the simulation. Run this like:
```
python step2_run_simulation.py 1e19_n1e3.hdf5 station.json config.yaml output.hdf5 output.nur
```

This will runt he actual simulation for those 1000 events. 
`station.json` defines the detector, and `config.yaml` specifies the simulation
settings (for things like ice model, physics models, etc).
`output.hdf5` and `output.nur` contain the output of the simulation.
`output.hdf5` contains relatively high level information about the events,
while `output.nur` contains simulated data, detailed MC truth information, etc.

### Step 3
Plot the results of our simulation. 
We are most interested in the waveforms (volts vs time).
Run like:
```
python step3_plot_traces.py
```
It should produce a series of png files showing waveforms.
The next step is to turn these into images for the CNN to usse.

## Fast Start

If you'd like to jump to make plots, then you can skip to step 3,
and use some "sample events" I've put here: `/i3store/users/rnog/sim_sample/`.
