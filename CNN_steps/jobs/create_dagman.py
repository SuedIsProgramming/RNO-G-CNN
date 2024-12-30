import argparse
import os
import numpy as np

#  Taken from https://github.com/umd-pa/tutorials-icecube/blob/main/convert_i3_to_hdf5/step2/step2a_make_dag.py

os.chdir('/data/i3home/ssued/RNOGCnn/CNN_steps/jobs')

sym_num = 2 # Number of simulations to run

# now we have to write the dag file itself
dag_filename = f"dagman.dag"
instructions = ""
#instructions += 'CONFIG config.dagman \n' Im not sure this is necessary

for i in range(1, sym_num+1):
    instructions += f'JOB step1_{i} step1.sub \n'
    instructions += f'JOB step2_{i} step2.sub \n'
    instructions += f'VARS step2_{i} inputfilename="1e19_n1e3.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output.hdf5" outputfilenameNuRadioReco="output.nur" \n'
    instructions += f'JOB step3_{i} step3.sub \n'
    instructions += f'JOB step4_{i} step4.sub \n\n'
    instructions += f'PARENT step1_{i} CHILD step2_{i}\n'
    instructions += f'PARENT step2_{i} CHILD step3_{i}\n'
    instructions += f'PARENT step3_{i} CHILD step4_{i}\n\n'

with open(dag_filename, 'w') as fwrite:
    fwrite.write(instructions)
