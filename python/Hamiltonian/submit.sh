#!/bin/bash
#
#SBATCH --job-name=kh
#SBATCH --output=data/res.txt

# parallel job using 128 processors. and runs for 4 hours (max)
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
# sends mail when process begins, and 
# when it ends. Make sure you define your email 
#SBATCH --mail-type=end
#SBATCH --mail-user=cnstahl@princeton.edu

module purge 
module load anaconda3

python L11end1n60_here.py
python L11end3n20_here.py
