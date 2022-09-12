#!/bin/bash

#SBATCH -J cigale
#SBATCH -q debug
#SBATCH -A dessn
#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --output=cigale_output.log

starttime=`date +%s`

source /global/common/software/dessn/cori_haswell/setup_cigale.sh

cd ${SLURM_SUBMIT_DIR}

echo STARTING CIGALE in ${SLURM_SUBMIT_DIR}

pcigale run
#pcigale-plots sed

endtime=`date +%s`
echo $endtime $starttime ${SLURM_SUBMIT_DIR} | awk '{printf "%100s  Runtime = %d seconds\n",$3,$1-$2}'
echo "JOBS FINISHED"
