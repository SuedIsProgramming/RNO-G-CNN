import numpy as np
import argparse
import sys
import os

#  Taken from https://github.com/umd-pa/tutorials-icecube/blob/main/convert_i3_to_hdf5/step2/step2a_make_dag.py

# This script creates a DAG file for the condor job scheduler. The DAG file is a text file that specifies the order in which jobs should be run.

os.chdir('/data/i3home/ssued/RNOGCnn/CNN_steps/jobs')

sym_num = 64 # Number of simulations to run 

# now we have to write the dag file itself
dag_filename = f"dagman.dag"
instructions = ""
#instructions += 'CONFIG config.dagman \n' Im not sure this is necessary

instructions += f'JOB stepOne_0 step1.sub \n'
instructions += f'VARS stepOne_0 sim_num="0" \n'
instructions += f'JOB stepTwo_0 step2.sub \n'
instructions += f'VARS stepTwo_0 inputfilename="1e19_n1e3.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output.hdf5" outputfilenameNuRadioReco="output.nur" \n'
instructions += f'JOB stepThree_0 step3.sub \n\n'
instructions += f'PARENT stepOne_0 CHILD stepTwo_0\n'
instructions += f'PARENT stepTwo_0 CHILD stepThree_0\n\n'

if sym_num == 1:
    instructions += f'JOB stepFour step4.sub \n'
    instructions += f'PARENT stepThree_0 CHILD stepFour'
    with open(dag_filename, 'w') as fwrite:
        fwrite.write(instructions)
    sys.exit()

sym_num = sym_num - 1 # shh, dont tell anyone

for i in range(1, sym_num):
    instructions += f'JOB stepOne_{i} step1.sub \n'
    instructions += f'VARS stepOne_{i} sim_num="{i}" \n'
    instructions += f'JOB stepTwo_{i} step2.sub \n'
    instructions += f'VARS stepTwo_{i} inputfilename="1e19_n1e3.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output.hdf5" outputfilenameNuRadioReco="output.nur" \n'
    instructions += f'JOB stepThree_{i} step3.sub \n\n'
    #instructions += f'PARENT stepThree_{i-1} CHILD stepOne_{i} \n'
    instructions += f'PARENT stepOne_{i} CHILD stepTwo_{i} \n'
    instructions += f'PARENT stepTwo_{i} CHILD stepThree_{i} \n\n'

instructions += f'JOB stepOne_{sym_num} step1.sub \n'
instructions += f'VARS stepOne_{sym_num} sim_num="{sym_num}" \n'
instructions += f'JOB stepTwo_{sym_num} step2.sub \n'
instructions += f'VARS stepTwo_{sym_num} inputfilename="1e19_n1e3.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output.hdf5" outputfilenameNuRadioReco="output.nur" \n'
instructions += f'JOB stepThree_{sym_num} step3.sub \n'
instructions += f'JOB stepFour step4.sub \n\n'
#instructions += f'PARENT stepThree_{sym_num-1} CHILD stepOne_{sym_num} \n'
instructions += f'PARENT stepOne_{sym_num} CHILD stepTwo_{sym_num}\n'
instructions += f'PARENT stepTwo_{sym_num} CHILD stepThree_{sym_num}\n\n'
for i in range(1, sym_num+1):
    instructions += f'PARENT stepThree_{i} CHILD stepFour\n'

with open(dag_filename, 'w') as fwrite:
    fwrite.write(instructions)
