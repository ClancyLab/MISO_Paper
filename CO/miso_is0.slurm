#!/bin/sh
#SBATCH --job-name="miso_is0"
#SBATCH --output="miso_is0.a%a.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00

#SBATCH --exclude=gpu004,gpu005
#SBATCH --array=0-200

date
module load pal
/software/apps/anaconda/5.2/python/3.7/bin/python -u /scratch/users/hherbol1@jhu.edu/miso_redo/CO/miso_is0.py > /scratch/users/hherbol1@jhu.edu/miso_redo/CO/miso_is0.log 2>&1
date

