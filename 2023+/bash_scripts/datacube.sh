

# module load python
# conda activate myenv
# echo $1 | tee -a output.txt
# python /pscratch/sd/j/joygong/Data-Cube-/afterprocessing.py afterprocessing $1 debug | tee -a output.txt

module load python
conda activate myenv
echo $1
python /pscratch/sd/j/joygong/Data-Cube-/afterprocessing.py afterprocessing $1 debug

