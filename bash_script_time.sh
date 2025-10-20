#!/bin/sh
### ------------- specify queue name ----------------
#BSUB -q c02516
### ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
### ------------- specify job name ----------------
#BSUB -J testjob
### ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=20GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 00:10
# Create folder
mkdir -p result

#BSUB -o result/earlyFusion%J.out
#BSUB -e result/earlyFusion%J.err

source venv_proj2/bin/activate
python early_fusion.py

timestamp=$(date +%Y%m%d-%H%M%S)
mv result/earlyFusion.out result/earlyFusion_${timestamp}.out
mv result/earlyFusion.err result/earlyFusion_${timestamp}.err
del result/earlyFusion.out
del result/earlyFusion.out
