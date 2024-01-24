# Simulating 3D Galaxy Datacubes for the ***Roman*** Space Telescope

## To simulate datacubes, follow these steps:
1. Download all files in the directories `necessary_files`,  `python_scripts`, and `bash_scripts`. In `afterprocessing.py` and `precigale_method1.py`, make sure to change the default path name to your local directory.
2. Make sure dependencies are installed (see python scripts), including [CIGALE]([url](https://cigale.lam.fr/)). Set up CIGALE in your directory and modify the source path in `runCIGALE.sh`.
3. To run locally, execute `./datacubes.sh` (you can modify the number of datacubes you simulate). 
4. To submit batch jobs, execute `./run_zs.sh` (again, you can modify the number of datacubes you simulate). Make sure to change `#SBATCH -q debug` to `#SBATCH -q regular` in `run_z.sl` if you are simulating > 5 datacubes. 
5. To plot results (both images and SEDs), use `Visualization_Spectra.ipynb` in the `visualization` directory.


