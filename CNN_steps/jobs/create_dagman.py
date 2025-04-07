import sys

#  Taken from https://github.com/umd-pa/tutorials-icecube/blob/main/convert_i3_to_hdf5/step2/step2a_make_dag.py

# This script creates a DAG file for the condor job scheduler. The DAG file is a text file that specifies the order in which jobs should be run.

# Change to directory of this script.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

sim_num = 2048*4 # Number of simulations to run 

# now we have to write the dag file itself
dag_filename = f"dagman.dag"
instructions = ""
#instructions += 'CONFIG config.dagman \n' Im not sure this is necessary

instructions += f'JOB stepOne_0 step1.sub \n'
instructions += f'VARS stepOne_0 sim_num="0" \n'
instructions += f'JOB stepTwo_0 step2.sub \n'
instructions += f'VARS stepTwo_0 sim_num="0" inputfilename="1e19_n1e3_0.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output_0.hdf5" outputfilenameNuRadioReco="output_0.nur" \n'
instructions += f'JOB stepThree_0 step3.sub \n'
instructions += f'VARS stepThree_0 sim_num="0" \n\n'
instructions += f'PARENT stepOne_0 CHILD stepTwo_0\n'
instructions += f'PARENT stepTwo_0 CHILD stepThree_0\n\n'

if sim_num == 1:
    instructions += f'JOB stepFour step4.sub \n'
    instructions += f'PARENT stepThree_0 CHILD stepFour'
    with open(dag_filename, 'w') as fwrite:
        fwrite.write(instructions)
    sys.exit()

sim_num = sim_num - 1 # shh, dont tell anyone

for i in range(1, sim_num):
    instructions += f'JOB stepOne_{i} step1.sub \n'
    instructions += f'VARS stepOne_{i} sim_num="{i}" \n'
    instructions += f'JOB stepTwo_{i} step2.sub \n'
    instructions += f'VARS stepTwo_{i} sim_num="{i}" inputfilename="1e19_n1e3_{i}.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output_{i}.hdf5" outputfilenameNuRadioReco="output_{i}.nur" \n'
    instructions += f'JOB stepThree_{i} step3.sub \n'
    instructions += f'VARS stepThree_{i} sim_num="{i}" \n\n'
    instructions += f'PARENT stepOne_{i} CHILD stepTwo_{i} \n'
    instructions += f'PARENT stepTwo_{i} CHILD stepThree_{i} \n\n'

instructions += f'JOB stepOne_{sim_num} step1.sub \n'
instructions += f'VARS stepOne_{sim_num} sim_num="{sim_num}" \n'
instructions += f'JOB stepTwo_{sim_num} step2.sub \n'
instructions += f'VARS stepTwo_{sim_num} sim_num="{sim_num}" inputfilename="1e19_n1e3_{sim_num}.hdf5" detectordescription="station.json" config="config.yaml" outputfilename="output_{sim_num}.hdf5" outputfilenameNuRadioReco="output_{sim_num}.nur" \n'
instructions += f'JOB stepThree_{sim_num} step3.sub \n'
instructions += f'VARS stepThree_{sim_num} sim_num="{sim_num}" \n\n'
instructions += f'PARENT stepOne_{sim_num} CHILD stepTwo_{sim_num}\n'
instructions += f'PARENT stepTwo_{sim_num} CHILD stepThree_{sim_num}\n\n'

instructions += 'FINAL stepFour step4.sub\n'

with open(dag_filename, 'w') as fwrite:
    fwrite.write(instructions)
