# HPC Instructions
This document describes how to use the DTU HPC for the scope of this project. 

## Access HPC
It is recommended to use the VS Code extension *Remote - SSH* by Microsoft. The extensions allow access to the HPC as an integrated part of the VS Code UI instead of in an external terminal.  

To access the HPC, choose the command `Remote-SSH: Connect to Host...` provided by the extension. 

Then parse in the following command: 
```
`ssh s123456@login1.hpc.dtu.dk`
```
and replace `123456` with your student ID. You will then be asked to type in your DTU password. After this has been done once, the extension will save the access point and it will show up as a suggestion.  

## Move Local Files to HPC
To move a file from your local machine to the HPC, use the following command from a new terminal in which you are not logged on to the HPC but can access your local files:
```
`scp <path_to_file_on_local_machine> s123456@login1.hpc.dtu.dk:<path_on_hpc>`
```
A path on the HPC to store your files could be `/zhome/xx/y/123456`. 

Remember that it is possible to open any folder on the HPC in VS Code through the Remote - SSH extension. 

## Virtual Environment
Select the Python version you want, and load the corresponding module. If you use versions older than 3.6, the instructions to create the virtual environment may differ.
```
module load python3/3.10.13
``` 

Then create a virtual environment. In the example below, the name of the environment is `venv_1`: 
```
python3 -m venv venv_proj2
```

Activate the environment with the command:
```
source venv_proj2/bin/activate
```

When the environment is active the shell notes this to the left of the cursor (`<venv_name>`). 

Install package with:
```
python -m pip install scikit-learn
```

All those packages will be installed locally under the directory:
```
venv_1/lib/python3.10/site-packages/
```

When you have finished using the environment, just deactivate it with the command: 
```
deactivate
```

All the information above about virtual environments on the DTU HPC has been taken from this webpage: https://www.hpc.dtu.dk/?page_id=3678.

## Run Code on Interactive Node
Interactive nodes can be used to run test scripts before submitting an actual batch job.  

The following nodes can be accessed with the following commands:
- HPC interactive node (Xeon CPU): `qrsh`
- Gbar node (Opteron CPU): `linuxsh`

Once accessed, run your script as you would in a regular terminal: `python <script_name>`.

## Run Code as Batch Job
The following is needed to submit and run a batch job:
- Python script
- Virtual environment
- Bash script

An example bash script named `bash_script.sh` is provided in this repository. In the bash script you can specify the:
- Queue name
- Job name
- Number of cores
- CPU memory requirements
- Wall-clock time (for how long can the job run, max 12 hours)
- Output directory for results and potential error messages
- Virtual environment
- Python script to run 

To submit a batch job, run the following command:
```
bsub -app c02516_1g.10gb < <name_of_bash_script>
```
Run the command `bstat` to check the status of your submitted job.  

If a memory error occurs, it might help to use one of the larger compute nodes:
- `c02516_2g.20gb`
- `c02516_4g.40gb`
Just replace the name of the compute node in the command for submitting a batch job above. 

## Documentation
For more information about the DTU HPC, visit https://www.hpc.dtu.dk/. 