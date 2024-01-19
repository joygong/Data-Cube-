# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: NERSC Python
#     language: python
#     name: python3
# ---

# +
# FINAL CODE SUMMING UP EVERYTHING -- Make Roman datacube at high redshift z2, based on low SDSS redshift z1 
# TERMINAL COMMAND: python afterprocessing.py afterprocessing z2 

import os
import sys
import time 
import random 
import string 
import subprocess
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.wcs import wcs
from astropy.wcs import WCS
from astropy.convolution import convolve, Gaussian2DKernel

    
# using random.choices() to generate random strings
res = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase +
                                  string.digits, k=20))
os.system('mkdir -p {} && cd {}'.format(res, res))
default = '/pscratch/sd/j/joygong/Data-Cube-'

#global pwd 
pwd = default + '/' + res 
print(pwd)
os.system('cd {}'.format(pwd))
#os.getcwd()
#os.system('cd {}'.format(pwd))

    
def afterprocessing(z2, debug=None): 
    
    pixscale = 0.396/0.025 
    z1_exact = z2/pixscale                                                    # lower sdss redshift to make data cube 
    #global z1
    z1 = (1 + np.random.normal(loc=0.0, scale=0.05, size=None))*z1_exact      # randomness (ratio doesn't have to be exactly 15.84) 
    print(z1)

    # python precigale_method1.py precigale 0.10 && sbatch runCIGALE.sl && python postcigale.py    
    # (have to wait a bit to run python postcigale.py (run first two, wait, then run postcigale) 
    # returns 1) gal_indices, 2) gal_pix_method1.in. 
    
    if sys.argv[3] == None: 
        os.system("python {}/precigale_method1.py precigale {} {} && cd {} && ./runCIGALE.sh && echo done".format(default, z1, pwd, pwd))
        # make cd ${SLURM_SUBMIT_DIR} take argument = SLURM_SUBMIT_DIR # (and then maybe make sbatch runCIGALE.sl [argument] 
        
    else: 
        os.system("python {}/precigale_method1.py precigale {} {} debug && cd {} && ./runCIGALE.sh && echo done".format(default, z1, pwd, pwd))

    print(pwd)
    output = subprocess.check_output("ls {} | grep gal_indices".format(pwd), shell=True).strip()
    str = output.decode('utf-8')
    print(str)
    
    gal_indices = pd.read_csv('{}/{}'.format(pwd, str))    
    redshift = gal_indices.iloc[0]['redshift']
    
    print(redshift) 
    
    length = 6 + len(gal_indices)*2
    print('length =', length)
    
    def number_of_files(path):     
        cmd = "ls {}/out | wc -l".format(path) 
        output = subprocess.run(cmd, capture_output=True, shell=True).stdout
        #output is a string
        return int(output)
        
#     while number_of_files('{}'.format(default)) < length:  # first check that all are in default/out 
#         time.sleep(1) 
#         os.system('echo checking original out files')
#         print('default current # = ', number_of_files('{}'.format(default)))
    
    # then copy over
    # if number_of_files('{}'.format(default)) == length: 
    #     os.system('cd {} && cp -r {}/out {} && echo done with copying!'.format(pwd, default, pwd))  # ACTUALLY THIS WOULDN'T WORK IF YOU'RE RUNNING FOR 1000S OF GALAXIES... 

    def checkexistence_out(): 
        cmd = "ls {}/out".format(pwd)     # in pwd
        status = subprocess.run(cmd, capture_output=True, shell=True).returncode
        return status
    
    while checkexistence_out() != 0: 
        time.sleep(1)
        #os.system('echo creating out directory') 

    if checkexistence_out() == 0:
        while number_of_files('{}'.format(pwd)) < length:
            time.sleep(1) 
            #os.system('echo generating out files')
            
    # run postcigale   
    os.system("python {}/postcigale.py postcigale {} && echo done".format(default, pwd))      
    
    # for future - may want to convert units of flux to erg/s/cm^2/A instead
    
    # open constructed sdss datacube @ z1
    # check that datacube file exists before attempting to open
    def checkexistence(): 
        cmd = "ls {}/datacube_{}.fits".format(pwd, redshift)     # in pwd
        status = subprocess.run(cmd, capture_output=True, shell=True).returncode
        #output is a string
        return status
    
    while checkexistence() != 0: 
        time.sleep(1)
        
    cube = fits.open('{}/datacube_{}.fits'.format(pwd, redshift))      # both the sdss and roman datacube are saved
    
    #### 1: shift wavelength and spectra (flux) <-- shifting flux = dimming pt. 1 i think?????? (maybe not, might be another factor of 1+z) 
    
    # wavelength: multiply by 1+z (lambda_obs = (1+z)*lambda_rest) 
    wavelengths = cube[1].data                                   # units: nm
    wavelengths = wavelengths.astype('float64')*((1+z2)/(1+z1))  # redshift back by z1, then redshift to z2 

    # flux: divide by 1+z (f_obs = f_rest / (1+z)) ; i think this takes care of one of the dimming factors too. jk 
    flambda = cube[0].data                                      # units: W/m²/nm
    flambda = flambda/((1+z2)/(1+z1))                           # redshift back by z1, then redshift to z2 
 

    #### 2: dim galaxy spectra according to lumnosity distance (LOOK AT DIVIDING BY ANOTHER FACTOR OF 1+Z!!!!!!!) 
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)                       # approximation; can import cosmo from other places e.g. wmap9 
    fluxfactor = (cosmo.luminosity_distance(z1).value/cosmo.luminosity_distance(z2).value)**2    # 01/05/24 update yeah there should be another factor of (1+z)??? wait nvm the flux already takes care of that i think uhh
    # source: https://www.physicsforums.com/threads/how-does-flux-density-scale-with-redshift.924414/
    flambda = flambda * fluxfactor                              # there might be another dimming factor of *(1+z1)/(1+z2)... 

    
    #### 2.5: rewrite stuff back into datacube (DO THIS FIRST BEFORE CHANGING HEADER)

    header = cube[0].header
    header['CUNIT1'] = ('pix     ', 'X units')           
    header['CUNIT2'] = ('pix     ', 'Y units')   
    header['CUNIT3'] = ('nm      ', 'Z / spectral units')     
    header['BUNIT'] = ('W/m^2/nm', '1 W/m^2/nm = 1000 erg/cm^2/s/nm = 10000 .../Ang')   # flux units
    
    datacube_hdu = fits.PrimaryHDU(flambda, header=header)        # unit = W/m²/nm
    wavelengths = wavelengths
    col = fits.Column(name='wavelengths', array=wavelengths, format='D', unit = 'nm')
    wavelengths_hdu = fits.BinTableHDU.from_columns([col])
    hdu_list = fits.HDUList([datacube_hdu, wavelengths_hdu])

    hdu_list.writeto('{}/roman_datacube_no_smooth_{}.fits'.format(pwd, z2), overwrite=True)

    # 3: resample to get smaller pixel scale for roman, incorporate angular resolution somehow? 
    # 4: have to reduce size of galaxy based on angular diameter distance (based on z) 
    #### 5 INSTEAD: JUST SELECT Z1 TO BE A Z2 MODEL + DO THE OTHER STUFF ACCORDINGLY... JUST CHANGE HEADER. 
        # Roman = 0.025" / pixel; same number of pixels (201x201) = bigger image of galaxy at same z b/c image is subtending a smaller total "
        # SDSS = 0.396" / pixel; same number of pixels (201x201) = smaller image of galaxy at same z b/c image is subtending bigger total "
        # So SDSS @ low redshift = same appearance/image as Roman @ high redshift if ratio is correct. (z2/z1 = 0.396/0.025 = 15.84) 
            # Q: Why does this link https://roman.gsfc.nasa.gov/science/WFI_technical.html say Roman is 0.11" / pixel ?
            # A: Have to oversample
    
    cube = fits.open('{}/roman_datacube_no_smooth_{}.fits'.format(pwd, z2))      

    header = cube[0].header
    scale = 0.025/3600               # Roman pixel scale (cdelt = pix increment in deg. 1" = 1/3600 deg)

    w = wcs.WCS(naxis=3)

    w.wcs.ctype = ['', '', '']
    w.wcs.crval = [0.0, 0.0, 0.0] 
    w.wcs.crpix = [0.0, 0.0, 0.0]
    w.wcs.cdelt = np.array([scale, scale, scale])

    cube[0].header.update(w.to_header())
    #cube[0].header
    #WCS(cube[0].header)

    cube.writeto('{}/roman_datacube_no_smooth_{}.fits'.format(pwd, z2), overwrite=True)
    cube.close()
    
  
    ###### smoothing w gaussian kernel - make another datacube in case want the original roman ######
    
    file = fits.open('{}/roman_datacube_no_smooth_{}.fits'.format(pwd, z2))
    
    flambda = file[0].data
    smoothed_datacube = np.empty_like(flambda)

    for i in range(len(flambda)): 
        sed = file[0].data[i, :, :]
        gauss_kernel = Gaussian2DKernel(1)       # sigma = 1 pixel (width)
        smoothed_data_gauss = convolve(sed, gauss_kernel, normalize_kernel=True)
        smoothed_datacube[i, :, :] = smoothed_data_gauss
    
    scale = 0.025/3600               # Roman oversampled pixel scale 
    w = wcs.WCS(naxis=3)

    w.wcs.ctype = ['', '', '']
    w.wcs.crval = [0.0, 0.0, 0.0] 
    w.wcs.crpix = [0.0, 0.0, 0.0]
    w.wcs.cdelt = np.array([scale, scale, scale])

    header = w.to_header()

    header['CUNIT1'] = ('pix     ', 'X units')           
    header['CUNIT2'] = ('pix     ', 'Y units')   
    header['CUNIT3'] = ('nm      ', 'Z / spectral units')     
    header['BUNIT'] = ('W/m^2/nm', '1 W/m^2/nm = 1000 erg/cm^2/s/nm = 10000 .../Ang')   # flux units
    
   
    datacube_hdu_smooth = fits.PrimaryHDU(smoothed_datacube, header=header)        #unit = W/m²/nm
    #wavelengths = fits.open('{}/roman_datacube_no_smooth_{}.fits'.format(pwd, z2))[1].data['wavelength']
    wavelengths = wavelengths
    col = fits.Column(name='wavelengths', array=wavelengths, format='D', unit = 'nm')
    wavelengths_hdu = fits.BinTableHDU.from_columns([col])
    hdu_list_smooth = fits.HDUList([datacube_hdu_smooth, wavelengths_hdu])
    
    
    hdu_list_smooth.writeto('{}/roman_datacube_{}.fits'.format(pwd, z2), overwrite=True)
    
    # append path of roman datacube to .txt file of all galaxy paths
    # with open("{}/datacube_paths.txt".format(default), "a") as f: # append mod
    # with open("{}/datacube_pathsNEW_vis.txt".format(default), "a") as f:     # for visualization (aas) 
    with open("{}/datacube_paths_011324_temp.txt".format(default), "a") as f:       # as of better? cleaning 01/13/24 
        f.write("{}/roman_datacube_{}.fits\n".format(pwd, z2))
        
    # directory names (easy for visualization purposes)
    with open("{}/dir_paths.txt".format(default), "a") as f:     # for visualization (aas) 
        f.write("{}\n".format(pwd))

    os.system('echo DONE')
    
if __name__ == '__main__':
    globals()[sys.argv[1]](float(sys.argv[2]), sys.argv[3])        #convert str input in terminal to float 
    
