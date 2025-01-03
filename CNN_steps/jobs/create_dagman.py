import numpy as np
import argparse
import sys
import os

#  Taken from https://github.com/umd-pa/tutorials-icecube/blob/main/convert_i3_to_hdf5/step2/step2a_make_dag.py

# This script creates a DAG file for the condor job scheduler. The DAG file is a text file that specifies the order in which jobs should be run.

os.chdir('/data/i3home/ssued/RNOGCnn/CNN_steps/jobs')

sym_num = 1 # Number of simulations to run 

# now we have to write the dag file itself
dag_filename = f"dagman.dag"
instructions = ""
#instructions += 'CONFIG config.dagman \n' Im not sure this is necessary

instructions += f'JOB step1_0 step1.sub \n'
instructions += f'JOB step2_0 step2.sub \n'
instructions += f'VARS step2_0 inputfilename="1e19_n1e3.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output.hdf5" outputfilenameNuRadioReco="output.nur" \n'
instructions += f'JOB step3_0 step3.sub \n\n'
instructions += f'PARENT step1_0 CHILD step2_0\n'
instructions += f'PARENT step2_0 CHILD step3_0\n\n'

if sym_num == 1:
    instructions += f'JOB step4 step4.sub \n'
    instructions += f'PARENT step3_0 CHILD step4'
    with open(dag_filename, 'w') as fwrite:
        fwrite.write(instructions)
    sys.exit()

sym_num = sym_num - 1 # shh, dont tell anyone

for i in range(1, sym_num):
    instructions += f'JOB step1_{i} step1.sub \n'
    instructions += f'JOB step2_{i} step2.sub \n'
    instructions += f'VARS step2_{i} inputfilename="1e19_n1e3.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output.hdf5" outputfilenameNuRadioReco="output.nur" \n'
    instructions += f'JOB step3_{i} step3.sub \n\n'
    instructions += f'PARENT step3_{i-1} CHILD step1_{i} \n'
    instructions += f'PARENT step1_{i} CHILD step2_{i} \n'
    instructions += f'PARENT step2_{i} CHILD step3_{i} \n\n'

instructions += f'JOB step1_{sym_num} step1.sub \n'
instructions += f'JOB step2_{sym_num} step2.sub \n'
instructions += f'VARS step2_{sym_num} inputfilename="1e19_n1e3.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output.hdf5" outputfilenameNuRadioReco="output.nur" \n'
instructions += f'JOB step3_{sym_num} step3.sub \n'
instructions += f'JOB step4 step4.sub \n\n'
instructions += f'PARENT step3_{sym_num-1} CHILD step1_{sym_num} \n'
instructions += f'PARENT step1_{sym_num} CHILD step2_{sym_num}\n'
instructions += f'PARENT step2_{sym_num} CHILD step3_{sym_num}\n'
instructions += f'PARENT step3_{sym_num} CHILD step4\n'

with open(dag_filename, 'w') as fwrite:
    fwrite.write(instructions)
