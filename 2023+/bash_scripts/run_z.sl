#!/bin/bash

#SBATCH -J roman
#SBATCH -q debug
#SBATCH -A dessn
#SBATCH -N 1
#SBATCH -t 0:10:00
#SBATCH -C cpu

module load python
conda activate myenv
python /pscratch/sd/j/joygong/Data-Cube-/afterprocessing.py afterprocessing $1 debug

