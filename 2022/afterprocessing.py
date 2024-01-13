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
import subprocess
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.wcs import wcs
from astropy.wcs import WCS


def afterprocessing(z2): 
    pixscale = 0.396/0.025 
    z1_exact = z2/pixscale                                                    # lower sdss redshift to make data cube 
    z1 = (1 + np.random.normal(loc=0.0, scale=0.05, size=None))*z1_exact      # randomness (ratio doesn't have to be exactly 15.84) 
    print(z1)

    # python precigale_method1.py precigale 0.10 && sbatch runCIGALE.sl && python postcigale.py    
    # (have to wait a bit to run python postcigale.py (run first two, wait, then run postcigale) 
    # returns 1) gal_indices, 2) gal_pix_method1.in. 
    os.system("python precigale_method1.py precigale {} && sbatch runCIGALE.sl && echo done".format(z1))

    gal_indices = pd.read_csv('gal_indices')  
    redshift = gal_indices.iloc[0]['redshift']
    print(redshift) 
    
    length = 6 + len(gal_indices)*2
    
    def number_of_files():
        cmd = "ls /global/cscratch1/sd/joygong/out | wc -l"
        output = subprocess.run(cmd, capture_output=True, shell=True).stdout
        #output is a string
        return int(output)
        
    while number_of_files() < length: 
        time.sleep(1) 
    
    # run postcigale   
    os.system("python postcigale.py && echo done")      
    
    # for future - may want to convert units of flux to erg/s/cm^2/A instead
    
    # open constructed sdss datacube @ z1
    # check that datacube file exists before attempting to open
    def checkexistence(): 
        cmd = "ls datacube_{}.fits".format(redshift)
        status = subprocess.run(cmd, capture_output=True, shell=True).returncode
        #output is a string
        return status
    
    while checkexistence() != 0: 
        time.sleep(1)
        
    cube = fits.open('datacube_{}.fits'.format(redshift))    
    
    #### 1: shift wavelength and spectra (flux) <-- shifting flux = dimming pt. 1 i think?????? (maybe not, might be another factor of 1+z) 
    
    # wavelength: multiply by 1+z (lambda_obs = (1+z)*lambda_rest) 
    wavelengths = cube[1].data                                   # units: nm
    wavelengths = wavelengths.astype('float64')*((1+z2)/(1+z1))  # redshift back by z1, then redshift to z2 

    # flux: divide by 1+z (f_obs = f_rest / (1+z)) ; i think this takes care of one of the dimming factors too. jk 
    flambda = cube[0].data                                      # units: W/m²/nm
    flambda = flambda/((1+z2)/(1+z1))                           # redshift back by z1, then redshift to z2 
 

    #### 2: dim galaxy spectra according to lumnosity distance (LOOK AT DIVIDING BY ANOTHER FACTOR OF 1+Z!!!!!!!) 
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)                       # approximation; can import cosmo from other places e.g. wmap9 
    fluxfactor = (cosmo.luminosity_distance(z1).value/cosmo.luminosity_distance(z2).value)**2
    flambda = flambda * fluxfactor                              # there might be another dimming factor of *(1+z1)/(1+z2)... 
    
    
    #### 2.5: rewrite stuff back into datacube (DO THIS FIRST BEFORE CHANGING HEADER)

    datacube_hdu = fits.PrimaryHDU(flambda)        # unit = W/m²/nm
    wavelengths = wavelengths
    col = fits.Column(name='wavelengths', array=wavelengths, format='D', unit = 'nm')
    wavelengths_hdu = fits.BinTableHDU.from_columns([col])
    hdu_list = fits.HDUList([datacube_hdu, wavelengths_hdu])

    hdu_list.writeto('roman_datacube_{}.fits'.format(z2), overwrite=True)

    # 3: resample to get smaller pixel scale for roman, incorporate angular resolution somehow? 
    # 4: have to reduce size of galaxy based on angular diameter distance (based on z) 
    #### 5 INSTEAD: JUST SELECT Z1 TO BE A Z2 MODEL + DO THE OTHER STUFF ACCORDINGLY... JUST CHANGE HEADER. 
        # Roman = 0.025" / pixel; same number of pixels (201x201) = bigger image of galaxy at same z b/c image is subtending a smaller total "
        # SDSS = 0.396" / pixel; same number of pixels (201x201) = smaller image of galaxy at same z b/c image is subtending bigger total "
        # So SDSS @ low redshift = same appearance/image as Roman @ high redshift if ratio is correct. (z2/z1 = 0.396/0.025 = 15.84) 
            # Q: Why does this link https://roman.gsfc.nasa.gov/science/WFI_technical.html say Roman is 0.11" / pixel ?
            # A: Have to oversample
    
    cube = fits.open('roman_datacube_{}.fits'.format(z2))      

    header = cube[0].header
    scale = 0.025/3600               # Roman pixel scale 

    w = wcs.WCS(naxis=3)

    w.wcs.ctype = ['', '', '']
    w.wcs.crval = [0.0, 0.0, 0.0] 
    w.wcs.crpix = [0.0, 0.0, 0.0]
    w.wcs.cdelt = np.array([scale, scale, scale])

    cube[0].header.update(w.to_header())
    #cube[0].header
    #WCS(cube[0].header)

    cube.writeto('roman_datacube_{}.fits'.format(z2), overwrite=True)
    cube.close()

    
if __name__ == '__main__':
    globals()[sys.argv[1]](float(sys.argv[2]))        #convert str input in terminal to float 
    
