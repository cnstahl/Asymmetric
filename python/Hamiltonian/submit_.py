#!/bin/bash
#
#SBATCH --job-name=version
#SBATCH --output=slurm_version.txt

# parallel job using 128 processors. and runs for 4 hours (max)
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH --mem=2000MB     
# sends mail when process begins, and 
# when it ends. Make sure you define your email 
#SBATCH --mail-type=end
#SBATCH --mail-user=cnstahl@princeton.edu

module purge

python3 version.py
