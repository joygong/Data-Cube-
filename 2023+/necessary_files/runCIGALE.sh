
starttime=`date +%s`

source /pscratch/sd/m/masao/setup_python.sh
conda activate cigale

# cd ${SLURM_SUBMIT_DIR}

# echo STARTING CIGALE in ${SLURM_SUBMIT_DIR}

pcigale run
#pcigale-plots sed

endtime=`date +%s`
# echo $endtime $starttime ${SLURM_SUBMIT_DIR} | awk '{printf "%100s  Runtime = %d seconds\n",$3,$1-$2}'
# echo "JOBS FINISHED"
