#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os 
import sys 
import subprocess
import numpy as np 
import pandas as pd
from astropy.io import fits 
from astropy.wcs import wcs
from astropy.wcs import WCS
from scipy.constants import c, pi
#note c = 3*10^8 m/s

def postcigale(pwd): 
    os.system("cd {}".format(pwd))
    print(pwd)
    # Put back spectra in the correct place w/ indices: use the same original data frame with id, x pix, y pix position. 

    data_cube = np.zeros([4291, 201, 201])     #3d array of all zeroes
    #wavelength_list = []                       #initialize wavelength list - same for all pixels...

    output = subprocess.check_output("ls {} | grep gal_indices".format(pwd), shell=True).strip()
    str = output.decode('utf-8')
    #str1 = str.replace('gal_indices_', '')

    gal_indices = pd.read_csv('{}/{}'.format(pwd, str))   # unique in each directory 
    redshift = gal_indices.iloc[0]['redshift']

    #pwd = os.getcwd()
    #pwd = afterprocessing.pwd

    # EACH PIXEL
    for i in range(0, len(gal_indices['id'])):                #len = 1224 total, range is excluding last one; also = len(sdss_r)

        def name(i): 
            name = '{}/out/{}_best_model.fits'.format(pwd, i)      # idk why it creates 2 out directories...
            return name  

        file = fits.open(name(i))
        fnu = file[1].data['Fnu']
        wavelength = file[1].data['wavelength']

        # Convert from Fnu to Flambda
        flambda = 1e-29 * 1e+9 * fnu / (wavelength * wavelength) * c      
        # ^ conversion taken from https://github.com/JohannesBuchner/cigale/blob/master/pcigale/sed/utils.py
            # input: fnu (mJy); wavelength (nm)
            # output: flam (W/m^2/nm)

        # Get pixel indices; data[y][x]
        y = gal_indices.loc[gal_indices['id'] == i, 'y_data_index'].values[0]
        x = gal_indices.loc[gal_indices['id'] == i, 'x_data_index'].values[0]

        # Change 1d array of 0's to flambda for each pixel 
        data_cube[:, y, x] = flambda
        #could add 3d array for wavelength too

    #data_cube

    # save to fits file with wavelength (one array, stored as bintable) 
    #note wavelength array is the same for all pixels -- bc it's just the x axes of the sed plot 

    scale = 0.396127/3600               # SDSS pixel scale (0.396"/pix, want in deg) [0.396127 arcsec/pix]             
                                        # source: https://skyserver.sdss.org/dr16/en/tools/chart/chartinfo.aspx
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
 
    datacube_hdu = fits.PrimaryHDU(data_cube, header=header)        #unit = W/m²/nm
    #wavelengths = fits.open('/global/cscratch1/sd/joygong/out/0_best_model.fits')[1].data['wavelength']
    wavelengths = fits.open('{}/out/0_best_model.fits'.format(pwd))[1].data['wavelength']
    col = fits.Column(name='wavelengths', array=wavelengths, format='D', unit = 'nm')
    wavelengths_hdu = fits.BinTableHDU.from_columns([col])
    hdu_list = fits.HDUList([datacube_hdu, wavelengths_hdu])

    hdu_list.writeto('{}/datacube_{}.fits'.format(pwd, redshift), overwrite=False)

    #datacube_hdu.writeto('datacube_test.fits')
    #hdu_list.writeto(f'datacube_reshaped_{os.path.basename(args.image_path)}.fits')
    #hdu_list.writeto(f'datacube_{os.path.basename(args.image_path)}.fits')

if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2]) 

    # In[ ]:
