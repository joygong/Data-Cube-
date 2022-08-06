#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from astropy.io import fits 
from scipy.constants import c, pi
#note c = 3*10^8 m/s


# Put back spectra in the correct place w/ indices: use the same original data frame with id, x pix, y pix position. 

data_cube = np.zeros([4654, 201, 201])     #3d array of all zeroes
#wavelength_list = []                       #initialize wavelength list - same for all pixels...

gal_indices = pd.read_csv('gal_indices')  
redshift = gal_indices.iloc[0]['redshift']

# EACH PIXEL
for i in range(0, len(gal_indices['id'])):                #len = 1224 total, range is excluding last one; also = len(sdss_r)
    
    def name(i): 
        name = '/global/cscratch1/sd/joygong/out/{}_best_model.fits'.format(i)
        return name  
    
    file = fits.open(name(i))
    fnu = file[1].data['Fnu']
    wavelength = file[1].data['wavelength']
    
    # Convert from Fnu to Flambda
    flambda = 1e-29 * 1e+9 * fnu / (wavelength * wavelength) * c
    
    # Get pixel indices; data[y][x]
    y = gal_indices.loc[gal_indices['id'] == i, 'y_data_index'].values[0]
    x = gal_indices.loc[gal_indices['id'] == i, 'x_data_index'].values[0]
    
    # Change 1d array of 0's to flambda for each pixel 
    data_cube[:, y, x] = flambda
    #could add 3d array for wavelength too

#data_cube

# save to fits file with wavelength (one array, stored as bintable) 
#note wavelength array is the same for all pixels -- bc it's just the x axes of the sed plot 

datacube_hdu = fits.PrimaryHDU(data_cube)        #unit = W/m²/nm
wavelengths = fits.open('/global/cscratch1/sd/joygong/out/0_best_model.fits')[1].data['wavelength']
col = fits.Column(name='wavelengths', array=wavelengths, format='D', unit = 'nm')
wavelengths_hdu = fits.BinTableHDU.from_columns([col])
hdu_list = fits.HDUList([datacube_hdu, wavelengths_hdu])

hdu_list.writeto('datacube_{}.fits'.format(redshift), overwrite=True)

#datacube_hdu.writeto('datacube_test.fits')
#hdu_list.writeto(f'datacube_reshaped_{os.path.basename(args.image_path)}.fits')
#hdu_list.writeto(f'datacube_{os.path.basename(args.image_path)}.fits')


# In[ ]:
