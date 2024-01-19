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
import os
import sys 
import string
import random
import numpy as np
import pandas as pd
import configparser 
from copy import deepcopy
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %matplotlib inline

from astropy.io import fits
from astropy.io import ascii
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.utils.data import download_file
from astropy.table import Table

from reproject import reproject_interp
from reproject import reproject_exact
from reproject.mosaicking import reproject_and_coadd
from reproject.mosaicking import find_optimal_celestial_wcs

from photutils.segmentation import detect_threshold
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
#from photutils.segmentation import make_source_mask
from photutils.segmentation import SourceCatalog
from photutils.background import Background2D
from photutils.background import MedianBackground

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.visualization import SqrtStretch
from astropy.visualization import ImageNormalize
from astropy.visualization import simple_norm
from astropy.visualization import AsinhStretch
from astropy.visualization import ZScaleInterval
from astropy.visualization import LogStretch
from astropy.stats import SigmaClip
from astropy.stats import gaussian_fwhm_to_sigma


###########
    
def precigale(input, pwd, debug=None): 
    
    print(pwd)
    os.system("cd {}".format(pwd))
    
    merged = pd.read_csv('/pscratch/sd/j/joygong/Data-Cube-/merged.csv')
    # subset = merged.head(100000)
    subset = merged     # if want to use all 

    if input < 100:
        if input > 0:        # pos redshift
            random = subset[(subset['Z'] > 0.95*input) & (subset['Z'] < 1.05*input)].sample(n=1)
        else:                # neg redshift
            random = subset[(subset['Z'] < 0.95*input) & (subset['Z'] > 1.05*input)].sample(n=1)

        run = random['RUN'].values[0]
        run6 = str(run).zfill(6)
        rerun = random['RERUN'].values[0]
        camcol = random['CAMCOL'].values[0]
        field = random['FIELD'].values[0]
        field4 = field4 = str(field).zfill(4)
        (ra,dec) = (random['RA'].values[0], random['DEC'].values[0])
        redshift = random['Z'].values[0]
        brightness = random['MAGR'].values[0]
    
    else: 
        run = subset.loc[subset['OBJID'] == input, 'RUN'].values[0]
        run6 = str(run).zfill(6)
        rerun = subset.loc[subset['OBJID'] == input, 'RERUN'].values[0]
        camcol = subset.loc[subset['OBJID'] == input, 'CAMCOL'].values[0]
        field = subset.loc[subset['OBJID'] == input, 'FIELD'].values[0]
        field4 = field4 = str(field).zfill(4)
        (ra,dec) = (subset.loc[subset['OBJID'] == input, 'RA'].values[0], subset.loc[subset['OBJID'] == input, 'DEC'].values[0])
        redshift = subset.loc[subset['OBJID'] == input, 'Z'].values[0]
        brightness = subset.loc[subset['OBJID'] == input, 'MAGR'].values[0] 
        
    # save galaxy parameters
    random.to_csv('{}/gal_params.csv'.format(pwd))
    print('brightness =', brightness)
    
    # URL function
    def construct(run, run6, rerun, camcol, field, field4, filter): 
        base = '{}/{}/{}/frame-{}-{}-{}-{}.fits.bz2'.format(rerun, run, camcol, filter, run6, camcol, field4)
        start = 'https://dr17.sdss.org/sas/dr17/eboss/photoObj/frames/'
        url = start + base
        return url

    urls = construct(run, run6, rerun, camcol, field, field4, 'u'), construct(run, run6, rerun, camcol, field, field4, 'g'), construct(run, run6, rerun, camcol, field, field4, 'r'), construct(run, run6, rerun, camcol, field, field4, 'i'), construct(run, run6, rerun, camcol, field, field4, 'z')
    print(ra,dec)
    
    file_u = download_file(urls[0])
    file_g = download_file(urls[1])
    file_r = download_file(urls[2])
    file_i = download_file(urls[3])
    file_z = download_file(urls[4])

    hdu0_u = fits.open(file_u)[0]
    hdu0_g = fits.open(file_g)[0]
    hdu0_r = fits.open(file_r)[0]
    hdu0_i = fits.open(file_i)[0]
    hdu0_z = fits.open(file_z)[0]
    
    # add argument to return original hdus if true & save to a directory 
    if sys.argv[4] != None: 
        
        u0 = hdu0_u.data
        g0 = hdu0_g.data
        r0 = hdu0_r.data
        i0 = hdu0_i.data
        z0 = hdu0_z.data

        hdu0s = np.stack( (u0,g0,r0,i0,z0) )

        hdu0s_fits = fits.PrimaryHDU(hdu0s)
        
        os.system('cd {} && mkdir -p debug/sdss_og'.format(pwd))
        path = '{}/debug/sdss_og'.format(pwd)
        hdu0s_fits.writeto('{}/hdu0s_{}.fits'.format(path, redshift), overwrite=False)

    
    ###### Resample/Cutout function for hdu0 ######
    def resample_cutout0(image): 
    
        # resample
        array, footprint = reproject_interp(image, hdu0_r.header)
        band_onto_r = fits.PrimaryHDU(array)

        # cutout
        wcs = WCS(hdu0_r.header) 
        center = wcs.all_world2pix(ra, dec, 0)
        size = 201
        cutout = Cutout2D(band_onto_r.data, center, size)

        return cutout.data

    
    cutout_r = resample_cutout0(hdu0_r)
    cutout_g = resample_cutout0(hdu0_g)
    cutout_u = resample_cutout0(hdu0_u)
    cutout_i = resample_cutout0(hdu0_i)
    cutout_z = resample_cutout0(hdu0_z)
    # print('cutout_u =', cutout_u)   # check data type 
    # print('cutout_u.data =', cutout_u.data)
        
    # argument to return resampled cutouts (or can do argparse...)
    if sys.argv[4] != None: 
      
        #pwd = os.getcwd()
        os.system('cd {} && mkdir -p debug/resampled'.format(pwd))
        #os.system('mkdir -p debug/resampled')
        path = '{}/debug/resampled'.format(pwd)
        #print(path)
        np.savez('{}/resampled_cutouts_{}'.format(path, redshift), cutout_r=cutout_r, cutout_g=cutout_g, cutout_u=cutout_u, cutout_i=cutout_i, cutout_z=cutout_z)
  
    
    
    ###### Removing other sources (individual bands)
    
    ### [1] ORIGINAL CLEAN FUNCTION ### 
#     def clean(data): 
#         #convolving/smoothing, segmenting, and deblending 
#         sigma_clip = SigmaClip(sigma=3.)
#         bkg_estimator = MedianBackground()
#         # bkg = Background2D(data, (20, 20), filter_size=(3, 3),                        #changed filter_size from (2,2) to (3,3) bc needs odd size for both axes apparently, also changed box size to (30,30).. tweak parameters for cleaning 
#         #                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)         #look into (10,10) and (2,2) aka was originally (50,50) and (3,3)
#         # bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator) 
#         bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
#         data -= bkg.background  # subtract the background
#         threshold = 2. * bkg.background_rms  # above the background

      
#         if brightness < 16.8: 
#             sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
#             kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)     # used to be 3
#             convolved_data = convolve(data, kernel, normalize_kernel=True)
#             npixels = 5
#             segm = detect_sources(convolved_data, threshold, npixels=npixels)
#             print(segm.data)
#             # maybe want to save segm data for debugging
#             segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
#                                        nlevels=32, contrast=0.15)

#         elif brightness >= 16.8 and brightness <= 17.1:  
#             # see if there's a more logical method to this range -- what if did mean + std, mean - std 
#                 # [17.77346101796728, 16.0427144022888, 16.90808771012804] <-- last is mean 
#             sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
#             kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
#             convolved_data = convolve(data, kernel, normalize_kernel=True)
#             npixels = 5
#             segm = detect_sources(convolved_data, threshold, npixels=npixels)
#             print(segm.data)
#             segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
#                                        nlevels=32, contrast=0.2)

#         elif brightness > 17.1: 
#             sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
#             kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
#             convolved_data = convolve(data, kernel, normalize_kernel=True)
#             npixels = 5
#             segm = detect_sources(convolved_data, threshold, npixels=npixels)
#             print(segm.data)
#             segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
#                                        nlevels=32, contrast=0.3)

#         # maybe have one more section for really dim galaxies --> even lower contrast


#         #table of (selected) properties for each source
#         cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
#         columns = ['label', 'xcentroid', 'ycentroid', 'area', 'bbox', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin']
#         tbl_data = cat.to_table(columns='data')
#         tbl = cat.to_table(columns=columns)
#         tbl['xcentroid'].info.format = '.4f'  # optional format
#         tbl['ycentroid'].info.format = '.4f'
#         #print('tbl =', tbl)   # QTable
#         print(tbl)
#         # write to table (saved)
#         if (data == cutout_u).all(): 
#             ascii.write(tbl, '{}/tbl_clean.ecsv'.format(pwd)) 
    
#         else: 
#             with open('{}/tbl_clean.ecsv'.format(pwd),'a') as f:
#                 tbl.write(f, format='ascii')

#         #assume (101,101) is central galaxy to isolate central source
#         galaxy = (tbl['xcentroid'] > 0.93*(data.shape[1]/2)) & (tbl['xcentroid'] < 1.07*(data.shape[1]/2)) & (tbl['ycentroid'] > 0.93*(data.shape[0]/2)) & (tbl['ycentroid'] < 1.07*(data.shape[0]/2))
#         #print(tbl[galaxy])   
#         n = tbl[galaxy][0]['label']         #get label/source number of central galaxy
            
#         # classify suspicious cleaning 
#         #tbl0 = tbl
#             # if the non-central source > 1/4th size of central source + is within [,] (15%?) of centroid, 
#             # then classify as suspicious. 
#             # 01/02/24 edited to 20% for safer bound, esp for spirals 
#         n_x = tbl[galaxy][0]['xcentroid']
#         n_y = tbl[galaxy][0]['ycentroid']
#         n_area = tbl[galaxy][0]['area']
        
#         suspicious = 0       #### a BETTER way would be to use the bbox....
#         for k in tbl['label']: 
#             if k != n: 
#                 if (0.80*n_x < tbl[k-1]['xcentroid'] < 1.20*n_x) and (0.80*n_y < tbl[k-1]['ycentroid'] < 1.20*n_y) and (tbl[k-1]['area'] > 0.25*n_area): 
#                     suspicious = 1
#                 else: 
#                     suspicious = 0
                
#         for k in tbl['label']:
#             xmin = tbl[k-1]['bbox_xmin']
#             xmax = tbl[k-1]['bbox_xmax']
#             ymin = tbl[k-1]['bbox_ymin']
#             ymax = tbl[k-1]['bbox_ymax']
#             #replace ALL sources with 0 first
#             data[ymin:ymax+1, xmin:xmax+1] = np.zeros((ymax+1 - ymin, xmax+1 - xmin), dtype=np.int64)

#         #print(cutout.data) 
#         #for bkg, select at random from any/all values that aren't = 0
#         #needs to be a flattened 1d array
#         bkg_cutout_flat = data.ravel()  
#         bkg_cutout_filtered = bkg_cutout_flat[np.nonzero(bkg_cutout_flat)]        #everything that is not 0's

#         #another loop to replace source_cutouts with background stuff (b/c has to iterate through all k's to replace all of those w 0)
#         for k in tbl['label']:
#             xmin = tbl[k-1]['bbox_xmin']
#             xmax = tbl[k-1]['bbox_xmax']
#             ymin = tbl[k-1]['bbox_ymin']
#             ymax = tbl[k-1]['bbox_ymax']

#             if k == n:   #replace central galaxy (set to 0) with original data
#                 data[ymin:ymax+1, xmin:xmax+1] = tbl_data[k-1]['data']           

#             else:        #replace other sources (now with 0's) with sky pixel values 
#                 data[ymin:ymax+1, xmin:xmax+1] = np.random.choice(bkg_cutout_filtered, size=(ymax+1 - ymin, xmax+1 - xmin))

#         data += bkg.background           #add background back
#         #plt.imshow(data, origin='lower')
#         # return data, tbl_clean, suspicious 
#         #return data, tbl0
#         return data, suspicious, segm, segm_deblend
    
    
    ##### [2] NEW (better) clean function as of 01/14/2024 #####
    def clean_modified(data): 
        data_og = np.copy(data)
        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        data -= bkg.background  # subtract the background
        threshold = 2. * bkg.background_rms  # above the background

        if brightness < 16.8: 
            sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
            kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)     # used to be 3
            convolved_data = convolve(data, kernel, normalize_kernel=True)
            npixels = 5
            segm = detect_sources(convolved_data, threshold, npixels=npixels)
            print(segm.data)
            # maybe want to save segm data for debugging
            segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
                                       nlevels=32, contrast=0.15)

        elif brightness >= 16.8 and brightness <= 17.1:  
            sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
            kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
            convolved_data = convolve(data, kernel, normalize_kernel=True)
            npixels = 5
            segm = detect_sources(convolved_data, threshold, npixels=npixels)
            print(segm.data)
            segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
                                       nlevels=32, contrast=0.2)

        elif brightness > 17.1: 
            sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
            kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
            convolved_data = convolve(data, kernel, normalize_kernel=True)
            npixels = 5
            segm = detect_sources(convolved_data, threshold, npixels=npixels)
            print(segm.data)
            segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
                                       nlevels=32, contrast=0.3)
            
       
        #table of (selected) properties for each source (NOTE! You actually don't need this for cleaning --> just to save) 
        cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
        columns = ['label', 'xcentroid', 'ycentroid', 'area', 'bbox', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin']
        tbl_data = cat.to_table(columns='data')
        tbl = cat.to_table(columns=columns)
        tbl['xcentroid'].info.format = '.4f'  # optional format
        tbl['ycentroid'].info.format = '.4f'
        print(tbl)
        # write to table (saved)

        if (data == cutout_u).all(): 
            ascii.write(tbl, '{}/tbl_clean.ecsv'.format(pwd)) 

        else: 
            with open('{}/tbl_clean.ecsv'.format(pwd),'a') as f:
                tbl.write(f, format='ascii')


        #assume (101,101) is central galaxy to isolate central source
        galaxy = (tbl['xcentroid'] > 0.93*(data.shape[1]/2)) & (tbl['xcentroid'] < 1.07*(data.shape[1]/2)) & (tbl['ycentroid'] > 0.93*(data.shape[0]/2)) & (tbl['ycentroid'] < 1.07*(data.shape[0]/2))
        #print(tbl[galaxy])   
        n = tbl[galaxy][0]['label']         #get label/source number of central galaxy

        # classify suspicious cleaning 
        n_x = tbl[galaxy][0]['xcentroid']
        n_y = tbl[galaxy][0]['ycentroid']
        n_area = tbl[galaxy][0]['area']

        suspicious = 0       #### a BETTER way would be to use the bbox....
        for k in tbl['label']: 
            if k != n: 
                if (0.80*n_x < tbl[k-1]['xcentroid'] < 1.20*n_x) and (0.80*n_y < tbl[k-1]['ycentroid'] < 1.20*n_y) and (tbl[k-1]['area'] > 0.25*n_area): 
                    suspicious = 1
                else: 
                    suspicious = 0

        # replace ALL sources with 0 first to get proper background estimation + data replacement later on 
        data[(segm_deblend.data > 0)] = 0      # note this used to be segm <-- wrong!! segm_deblend gets ALL connected sources too 

        #for bkg, select at random from any/all values that aren't = 0
        #needs to be a flattened 1d array
        bkg_cutout_flat = data.ravel()  
        bkg_cutout_filtered = bkg_cutout_flat[np.nonzero(bkg_cutout_flat)]        #everything that is not 0's

        # non-central data: replace with background pixels 
        data[(segm_deblend.data > 0) & (segm_deblend.data != n)] = np.random.choice(bkg_cutout_filtered, size=len(data[(segm_deblend.data > 0) & (segm_deblend.data != n)]))

        # replacement of central data
        data[(segm_deblend.data == n)] = data_og[(segm_deblend.data == n)]

        data += bkg.background           # add background back

    #     segm_og = np.copy(segm)          # original with ALL sources (b/c for some reason you can't keep label on the copy)
    #     segm_cleaned = segm.keep_label(label=n)   # note segm is the post-cleaned segmentation image 

        segm_deblend_og = np.copy(segm_deblend)
        segm_deblend_cleaned = segm_deblend.keep_label(label=n)  

        # return data, suspicious, segm_og, segm, segm_deblend_og, segm_deblend, n
        return data, suspicious, segm_deblend_og, segm_deblend, n


   ###############
       # here maybe make the contrast / other source detection parameters variables, so that can change based on band.
    
    # output (NOTE!! YOU CAN ONLY RUN CLEAN FUNCTION ONCE, BECAUSE DATA IS MODIFIED! otherwise source detection will go through EXTRA rounds of cleaning.)
    # cl_u = clean(cutout_u)
    # cl_g = clean(cutout_g)
    # cl_r = clean(cutout_r)
    # cl_i = clean(cutout_i)
    # cl_z = clean(cutout_z)
    
    cl_u = clean_modified(cutout_u)
    cl_g = clean_modified(cutout_g)
    cl_r = clean_modified(cutout_r)
    cl_i = clean_modified(cutout_i)
    cl_z = clean_modified(cutout_z)
    
    # take note of suspicious cleaning
    sus = cl_u[1] + cl_g[1] + cl_r[1] + cl_i[1] + cl_z[1]
    print('sus =', sus)
    
    #if sus > 0:    # maybe if more than 1 since u is shitty
    if sus > 1: 
        print("suspicious") 
        default = '/pscratch/sd/j/joygong/Data-Cube-'
        with open("{}/datacube_suspicious.txt".format(default), "a") as f: 
            f.write("{}\n".format(pwd))       # original unsmoothed datacube for sdss (how do i get the redshift of roman here...?) but visually they look the same when plotted
                # actually just the directory might be more useful
            #f.write("{}/datacube_{}\n".format(pwd, redshift)) 
        
    # if any of clean(cutout_u, g, r, i, z)[2] = 'yes': 
    #     save redshift, name etc. to file --> how to get outside redshift?? (i.e. pwd?) or can just get pwd
    #     + figure out the name of the roman file after
    
    # saving segmentation images 
#     segm_u = cl_u[2]     # THIS IS FOR THE ORIGINAL CLEAN FUNCTION
#     segm_g = cl_g[2]
#     segm_r = cl_r[2]
#     segm_i = cl_i[2]
#     segm_z = cl_z[2]
    
    # segm_u_d = cl_u[3]    # deblended ones 
    # segm_g_d = cl_g[3]
    # segm_r_d = cl_r[3]
    # segm_i_d = cl_i[3]
    # segm_z_d = cl_z[3]
    
    segm_u_og = cl_u[2]    # THIS IS FOR THE MODIFIED CLEAN FUNCTION. DEBLENDED segmentation image PRE-cleaning. includes all sources.
    segm_g_og = cl_g[2]
    segm_r_og = cl_r[2]
    segm_i_og = cl_i[2]
    segm_z_og = cl_z[2]
    
    segm_u_c = cl_u[3]    # THIS IS FOR THE MODIFIED CLEAN FUNCTION. segmentation image POST-cleaning. includes ONLY central source.
    segm_g_c = cl_g[3]
    segm_r_c = cl_r[3]
    segm_i_c = cl_i[3]
    segm_z_c = cl_z[3]
    
#     segms = np.stack( (segm_u, segm_g, segm_r, segm_i, segm_z) )
#     segms_d = np.stack( (segm_u_d, segm_g_d, segm_r_d, segm_i_d, segm_z_d) )

    segms_og = np.stack( (segm_u_og, segm_g_og, segm_r_og, segm_i_og, segm_z_og) )
    segms_cl = np.stack( (segm_u_c, segm_g_c, segm_r_c, segm_i_c, segm_z_c) )
        
    if sys.argv[4] != None: 
        
        os.system('cd {} && mkdir -p debug/segm'.format(pwd))
        path = '{}/debug/segm'.format(pwd)        
        # np.savez('{}/segm_{}'.format(path, redshift), segm_u=segm_u, segm_g=segm_g, segm_r=segm_r, segm_i=segm_i, segm_z=segm_z)
        np.savez('{}/segms_og_{}'.format(path, redshift), segms_og=segms_og)
        np.savez('{}/segms_cl_{}'.format(path, redshift), segms_cl=segms_cl)

    
    # pixel fluxes for cleaned data
    clean_u = cl_u[0]        # problematic sometimes bc can't detect a source; use g band detection for u (for higher redshifts usually) 
    clean_g = cl_g[0]
    clean_r = cl_r[0]
    clean_i = cl_i[0]
    clean_z = cl_z[0]

    # data to put into fits file w/ primaryhdu = 5 x 201 x 201
    u = clean_u.data
    g = clean_g.data
    r = clean_r.data
    i = clean_i.data
    z = clean_z.data

    clean = np.stack( (u,g,r,i,z) )
    #clean.shape

    clean_fits = fits.PrimaryHDU(clean)
    images = clean_fits.data

    # save clean_fits to directory if debug=True 
    if sys.argv[4] != None: 
        
        #pwd = os.getcwd()
        os.system('cd {} && mkdir -p debug/cleaned'.format(pwd))
        path = '{}/debug/cleaned'.format(pwd)
        clean_fits.writeto('{}/cleaned_{}.fits'.format(path, redshift), overwrite=False)
        

    ###### Filter Observation Map ######

    #### [1] FROM TRI: #### (THIS ONE CLEANS IT AGAIN AND LEAVES PIX THAT AREN'T PART OF THE GALAXY!!!) 
    ## The filter observation map is a 2d array with the same dimension as the images.
#     FILTER_NAMES  = ['u', 'g', 'r', 'i', 'z']

#     nCols = 1
#     nRows = 5

#     xSize = 15
#     ySize = xSize*float(nRows)/float(nCols)

#     wspace = 0.0
#     hspace = 0.0

#     #def_plot_values_extra_large()
#     fig = plt.figure(figsize=(xSize, ySize))
#     gs = gridspec.GridSpec(nrows=nRows, ncols=nCols, wspace=wspace, hspace=hspace)

#     ########

#     filterObsMap = np.zeros_like(images[0])
    
    #Plot sources (of cleaned images)
#     for i in range(nRows):
#         for j in range(nCols):
#             k  = i*nCols + j
        
#             if (k < 5):
#                 ax = plt.subplot(gs[k])
        
#                 text = FILTER_NAMES[k]+" Borders"
            
#                 ax.text(0.03, 0.96, text, color='white', ha='left', va='top', transform=plt.gca().transAxes)
            
#                 data = deepcopy(images[k])

#                 sigma_clip = SigmaClip(sigma=3.)
#                 bkg_estimator = MedianBackground()
#                 # bkg = Background2D(data, (20, 20), filter_size=(3, 3),
#                 #                 sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)         #look into (10,10) and (2,2) aka was originally (50,50) and (3,3)
#                 bkg = Background2D(data, (50, 50), filter_size=(5, 5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
#                 data -= bkg.background  # subtract the background
#                 threshold = 2. * bkg.background_rms  # above the background

#                 #convolving/smoothing, segmenting, and deblending 
#                 if brightness < 16.8: 
#                     sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
#                     kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
#                     convolved_data = convolve(data, kernel, normalize_kernel=True)
#                     npixels = 5
#                     segm = detect_sources(convolved_data, threshold, npixels=npixels)
#                     segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
#                                                nlevels=32, contrast=0.15)

#                 elif brightness >= 16.8 and brightness <= 17.1:  
#                     sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
#                     kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
#                     convolved_data = convolve(data, kernel, normalize_kernel=True)
#                     npixels = 5
#                     segm = detect_sources(convolved_data, threshold, npixels=npixels)
#                     segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
#                                                nlevels=32, contrast=0.2)

#                 elif brightness > 17.1: 
#                     sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
#                     kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
#                     convolved_data = convolve(data, kernel, normalize_kernel=True)
#                     npixels = 5
#                     segm = detect_sources(convolved_data, threshold, npixels=npixels)
#                     segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
#                                                nlevels=32, contrast=0.3)
            
#                 #table of (selected) properties for each source
#                 cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
#                 columns = ['label', 'xcentroid', 'ycentroid', 'area', 'bbox', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin']
#                 tbl_data = cat.to_table(columns='data')
#                 tbl = cat.to_table(columns=columns)
#                 tbl['xcentroid'].info.format = '.4f'  # optional format
#                 tbl['ycentroid'].info.format = '.4f'
    
#                 data += bkg.background           #add background back

#                 print(FILTER_NAMES[k], segm.areas, tbl)   # segm.background_area,
#                 #np.savez('{}/tbl_filter_obs_map'.format(pwd), tbl=FILTER_NAMES[k], segm.areas, tbl)
#                 # with open('{}/tbl_filter_obs_map.txt'.format(pwd), 'w') as f:
#                 #     #f.write(tabulate(table))
#                 #     #f.write(tbl) 
#                 #     f.write(FILTER_NAMES[k], segm.areas, tbl)
#                 #tbl.write('{}/tbl_filter_obs_map.ecsv'.format(pwd))
#                 if k == 0: 
#                     ascii.write(tbl, '{}/tbl_filter_obs_map.ecsv'.format(pwd))  
#                 else: 
#                     with open('{}/tbl_filter_obs_map.ecsv'.format(pwd),'a') as f:
#                         tbl.write(f, format='ascii')
        
#                 outline = segm.outline_segments()            
            
#                 data[outline > 0] = np.inf
                                    
#                 ## ax.imshow(masked, cmap = 'Reds')
            
#                 #ax.imshow(data, origin='lower', aspect='equal', vmin=vmin, vmax=vmax, interpolation='nearest', cmap='gray')
            
#                 ax.imshow(data, origin='lower', aspect='equal', interpolation='nearest', cmap='gray')       #comment out if don't want to see 

#                 filterObsMap[segm.data > 0] += 1.0
            
#                 if (i < (nRows-1)):
#                     ax.set_xticklabels('')
#                 if (j > 0):
#                     ax.set_yticklabels('')

#     xLabel, yLabel = r'$X$ [pix]', r'$Y$ [pix]'

    #axCommons = drawCommonLabel(xLabel, yLabel, fig, xPad=20, yPad=25)

    
    # Plot filter observation map
#     xSize = 10
#     ySize = xSize

#     #def_plot_values_extra_large()
#     fig = plt.figure(figsize=(xSize, ySize))

#     ax = plt.subplot(111)

#     im = ax.imshow(filterObsMap, cmap='tab20b', origin='lower', aspect='equal')      #comment out if don't want to see

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)

#     cbar = plt.colorbar(im, cax=cax)
#     cbar.set_label('Number of filter observations')

#     ax.set_xlabel(xLabel)
#     ax.set_ylabel(yLabel)
    
    # save for debugging 
#     if sys.argv[4] != None: 
        
#         #pwd = os.getcwd()
#         os.system('cd {} && mkdir -p debug/filter_obs_map'.format(pwd))
#         path = '{}/debug/filter_obs_map'.format(pwd)
#         np.savez('{}/filter_obs_map_{}'.format(path, redshift), filter_obs_map=filterObsMap)

        
#     # Nonzero indices
#     print(filterObsMap[filterObsMap >= 1].size, filterObsMap.size)       # prints number of pixels observed at least once, total pixels                      
#     nonZeroIndices = np.argwhere(filterObsMap >= 1.0)                    # assign 0 flux to non-source pixels        
#     print(nonZeroIndices.shape)                                          # shape 
#     print(nonZeroIndices)   # indices -- x y or y x? --> y x already


#     # Make dataframe of id, y index, x index
#     y_indices = np.hsplit(nonZeroIndices, 2)[0]     # this is y_indices; argwhere already puts into correct form (so left right is [y][x]) 
#     x_indices = np.hsplit(nonZeroIndices, 2)[1]
#     #print(x_indices, y_indices)    
    
#     y_list = y_indices.flatten
#     x_list = x_indices.flatten

#     gal_indices = pd.DataFrame()
#     gal_indices['y_data_index'] = y_list()
#     gal_indices['x_data_index'] = x_list()          # pix index in image; so reference it to be data[y][x]
#     gal_indices['redshift'] = [redshift] * len(y_indices)
#     gal_indices.reset_index(inplace=True)
#     gal_indices = gal_indices.rename(columns = {'index':'id'})

#     gal_indices      #index=False

    
#     #pwd = os.getcwd()
#     gal_indices.to_csv('{}/gal_indices_{}'.format(pwd, redshift), index=False)
    
    
    #### [2] THIS IS MY VERSION WHICH IS BETTER ####

    filterObsMap = np.zeros((201,201))
    filterObsMap[segm_u_c.data > 0] += 1.0    
    filterObsMap[segm_g_c.data > 0] += 1.0 
    filterObsMap[segm_r_c.data > 0] += 1.0
    filterObsMap[segm_i_c.data > 0] += 1.0
    filterObsMap[segm_z_c.data > 0] += 1.0

    xLabel, yLabel = r'$X$ [pix]', r'$Y$ [pix]'

    xSize = 15
    ySize = xSize

    #def_plot_values_extra_large()
    fig = plt.figure(figsize=(xSize, ySize))

    ax = plt.subplot(111)

    im = ax.imshow(filterObsMap, cmap='tab20b', origin='lower', aspect='equal')      #comment out if don't want to see

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Number of filter observations')

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    # save for debugging 
    if sys.argv[4] != None: 

        #pwd = os.getcwd()
        os.system('cd {} && mkdir -p debug/filter_obs_map'.format(pwd))
        path = '{}/debug/filter_obs_map'.format(pwd)
        np.savez('{}/filter_obs_map_{}'.format(path, redshift), filter_obs_map=filterObsMap)


    print(filterObsMap[filterObsMap >= 1].size, filterObsMap.size)       # prints number of pixels observed at least once, total pixels                      
    nonZeroIndices = np.argwhere(filterObsMap >= 1.0)                    # assign 0 flux to non-source pixels        
    print(nonZeroIndices.shape)                                          # shape 
    print(nonZeroIndices)   # indices -- x y or y x? --> y x already


    # Make dataframe of id, y index, x index
    y_indices = np.hsplit(nonZeroIndices, 2)[0]     # this is y_indices; argwhere already puts into correct form (so left right is [y][x]) 
    x_indices = np.hsplit(nonZeroIndices, 2)[1]
    #print(x_indices, y_indices)    

    y_list = y_indices.flatten
    x_list = x_indices.flatten

    gal_indices = pd.DataFrame()
    gal_indices['y_data_index'] = y_list()
    gal_indices['x_data_index'] = x_list()          # pix index in image; so reference it to be data[y][x]
    gal_indices['redshift'] = [redshift] * len(y_indices)
    gal_indices.reset_index(inplace=True)
    gal_indices = gal_indices.rename(columns = {'index':'id'})

    gal_indices      #index=False

    gal_indices.to_csv('{}/gal_indices_{}'.format(pwd, redshift), index=False)

    

    ######## Generate noise map ########

    # bilinear resize function 
    def bilinear_resize_vectorized(image, height, width):
      """
      `image` is a 2-D numpy array
      `height` and `width` are the desired spatial dimension of the new 2-D array.
      """
      img_height, img_width = image.shape

      image = image.ravel()

      x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
      y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

      y, x = np.divmod(np.arange(height * width), width)

      x_l = np.floor(x_ratio * x).astype('int32')
      y_l = np.floor(y_ratio * y).astype('int32')
    
      x_h = np.ceil(x_ratio * x).astype('int32')
      y_h = np.ceil(y_ratio * y).astype('int32')

      x_weight = (x_ratio * x) - x_l
      y_weight = (y_ratio * y) - y_l

      a = image[y_l * img_width + x_l]
      b = image[y_l * img_width + x_h]
      c = image[y_h * img_width + x_l]
      d = image[y_h * img_width + x_h]

      resized = a * (1 - x_weight) * (1 - y_weight) + \
                b * x_weight * (1 - y_weight) + \
                c * y_weight * (1 - x_weight) + \
                d * x_weight * y_weight

      return resized.reshape(height, width)



    # gain_darkVariance function 
    def gain_darkVariance(camcol, filter):      #note also dependent on run (needs to be correct; also filter should be entered w/ ' ')
    
        if camcol == 1: 
            if filter == 'u':
                gain = 1.62
                darkVariance = 9.61
            elif filter == 'g':
                gain = 3.32
                darkVariance = 15.6025
            elif filter == 'r':
                gain = 4.71
                darkVariance = 1.8225
            elif filter == 'i':
                gain = 5.165
                darkVariance = 7.84
            else:       # filter == 'z'
                gain = 4.745
                darkVariance = 0.81

        elif camcol == 2: 
            if filter == 'u':
                if run < 1100: 
                    gain = 1.595
                else: 
                    gain = 1.825
                darkVariance = 12.6025
            
            elif filter == 'g':
                gain = 3.855
                darkVariance = 1.44
            elif filter == 'r':
                gain = 4.6
                darkVariance = 1.00

            elif filter == 'i':
                gain = 6.565
                if run < 1500: 
                    darkVariance = 5.76
                else: 
                    darkVariance = 6.25
            
            else:       # filter == 'z'
                gain = 5.155
                darkVariance = 1.0

        elif camcol == 3: 
            if filter == 'u':
                gain = 1.59
                darkVariance = 8.7025
            elif filter == 'g':
                gain = 3.845
                darkVariance = 1.3225
            elif filter == 'r':
                gain = 4.72
                darkVariance = 1.3225
            elif filter == 'i':
                gain = 4.86
                darkVariance = 4.6225
            else:       # filter == 'z'
                gain = 4.885
                darkVariance = 1.0
         
        elif camcol == 4: 
            if filter == 'u':
                gain = 1.6
                darkVariance = 12.6025
            elif filter == 'g':
                gain = 3.995
                darkVariance = 1.96
            elif filter == 'r':
                gain = 4.76
                darkVariance = 1.3225
            elif filter == 'i':
                gain = 4.885
                if run < 1500 :
                    darkVariance = 6.25
                else: 
                    darkVariance = 7.5625
            else:       # filter == 'z'
                gain = 4.775
                if run < 1500 :
                    darkVariance = 9.61
                else: 
                    darkVariance = 12.6025
   
        elif camcol == 5: 
            if filter == 'u':
                gain = 1.47
                darkVariance = 9.3025
            elif filter == 'g':
                gain = 4.05	
                darkVariance = 1.1025
            elif filter == 'r':
                gain = 4.725
                darkVariance = 0.81
            elif filter == 'i':
                gain = 4.64
                darkVariance = 7.84
            else:       # filter == 'z'
                gain = 3.48
                if run < 1500: 
                    darkvariance = 1.8225
                else: 
                    darkVariance = 2.1025
            
        else:   # camcol == 6: 
            if filter == 'u':
                gain =  2.17
                darkVariance = 7.0225
            elif filter == 'g':
                gain = 4.035
                darkVariance = 1.8225
            elif filter == 'r':
                gain = 4.895
                darkVariance = 0.9025
            elif filter == 'i':
                gain = 4.76
                darkVariance = 5.0625
            else:       # filter == 'z'
                gain = 4.69
                darkVariance = 1.21
    
        return (gain, darkVariance)

    
    # function to generate noise map for any band 
    def noise(filter): 
        if filter == 'u': 
            hdu0 = hdu0_u
            hdu1 = fits.open(file_u)[1]
            hdu2 = fits.open(file_u)[2]
        elif filter == 'g':
            hdu0 = hdu0_g
            hdu1 = fits.open(file_g)[1]
            hdu2 = fits.open(file_g)[2]
        elif filter == 'r':
            hdu0 = hdu0_r
            hdu1 = fits.open(file_r)[1]
            hdu2 = fits.open(file_r)[2]
        elif filter == 'i':
            hdu0 = hdu0_i
            hdu1 = fits.open(file_i)[1]
            hdu2 = fits.open(file_i)[2]
        else:       #filter = 'z'
            hdu0 = hdu0_z
            hdu1 = fits.open(file_z)[1]
            hdu2 = fits.open(file_z)[2]
        
        x = hdu2.data['XINTERP']
        y = hdu2.data['YINTERP']
        z = hdu2.data['ALLSKY']
        new_z = np.reshape(z, (192,256), order='C')
    
        #simg
        simg = bilinear_resize_vectorized(new_z, 1489, 2048)

        #cimg
        nrowc = 1489
        cimg = np.tile(hdu1.data, (nrowc, 1))

        #img_err
        img = hdu0.data
        dn = img/cimg + simg
        gain, darkVariance = gain_darkVariance(camcol, filter)
        dn_err = np.sqrt(dn/gain + darkVariance)
        img_err = dn_err*cimg

        # save to fits file 
        err = fits.PrimaryHDU(img_err)
    
        return err

    err_u = noise('u') 
    err_g = noise('g') 
    err_r = noise('r') 
    err_i = noise('i') 
    err_z = noise('z') 

    
    # FUNCTION TO RESAMPLE EACH ONE (ONTO R)
    def resample_cutout(image): 
    
        # resample
        array, footprint = reproject_interp(image, err_r.header)
        err_onto_r = fits.PrimaryHDU(array)

        # cutout
        wcs = WCS(hdu0_r.header) 
        center = wcs.all_world2pix(ra, dec, 0)
        size = 201
        cutout_err = Cutout2D(err_onto_r.data, center, size)

        return cutout_err.data


    # Noise map cutouts
    u_err = resample_cutout(err_u)
    g_err = resample_cutout(err_g)
    r_err = resample_cutout(err_r)
    i_err = resample_cutout(err_i)
    z_err = resample_cutout(err_z)

    
    ######## Generate .in file ########
    
    # empty lists for all bands 
    sdss_u = []
    sdss_u_err = []  
    sdss_g = []
    sdss_g_err = []
    sdss_r = []
    sdss_r_err = []
    sdss_i = []
    sdss_i_err = []
    sdss_z = []
    sdss_z_err = []


    for i in gal_indices['id']: 
        y = gal_indices.loc[gal_indices['id'] == i, 'y_data_index'].values[0]
        x = gal_indices.loc[gal_indices['id'] == i, 'x_data_index'].values[0]
    
        flux_u = clean[0][y][x] * 3.631                         # in terms of mJy    
        flux_u_err = u_err[y][x] * 3.631                        # noise map stuff
        flux_g = clean[1][y][x] * 3.631 
        flux_g_err = g_err[y][x] * 3.631 
        flux_r = clean[2][y][x] * 3.631 
        flux_r_err = r_err[y][x] * 3.631 
        flux_i = clean[3][y][x] * 3.631 
        flux_i_err = i_err[y][x] * 3.631 
        flux_z = clean[4][y][x] * 3.631 
        flux_z_err = z_err[y][x] * 3.631         # flux_z_err = resample_cutout(err_z)[y][x] * 3.631    #to be safe
 

        sdss_u.append(flux_u)
        sdss_u_err.append(flux_u_err)
        sdss_g.append(flux_g)
        sdss_g_err.append(flux_g_err)
        sdss_r.append(flux_r)
        sdss_r_err.append(flux_r_err)
        sdss_i.append(flux_i)
        sdss_i_err.append(flux_i_err)
        sdss_z.append(flux_z)
        sdss_z_err.append(flux_z_err)

    # Construct table with id, redshift, sdss_u, sdss_u_err, etc. as columns (should only be galaxy pixels now) 
    # Generate ASCII table 
    redshift_full = np.full(len(gal_indices['id']), redshift)

    gal = pd.DataFrame()
    gal['redshift'] = redshift_full.tolist()
    gal['sdss.up'] = sdss_u
    gal['sdss.up_err'] = sdss_u_err
    gal['sdss.gp'] = sdss_g
    gal['sdss.gp_err'] = sdss_g_err
    gal['sdss.rp'] = sdss_r
    gal['sdss.rp_err'] = sdss_r_err
    gal['sdss.ip'] = sdss_i
    gal['sdss.ip_err'] = sdss_i_err
    gal['sdss.zp'] = sdss_z
    gal['sdss.zp_err'] = sdss_z_err
    gal.reset_index(inplace=True)
    gal = gal.rename(columns = {'index':'#id'})
    gal = Table.from_pandas(gal)
    #gal
    
    #pwd = os.getcwd()
    gal = ascii.write(gal, '{}/gal_pix_method1_{}.in'.format(pwd, redshift), overwrite=False)
    
    print(gal) 
    
    # copy over necessary files for cigale
    #pwd = os.getcwd()
    default = '/pscratch/sd/j/joygong/Data-Cube-'
    os.system('cp {}/pcigale.ini {}/pcigale.ini.spec {}/runCIGALE.sh {}/'.format(default, default, default, pwd))
              
    # modify pcigale.ini file to have input file = specific
    with open('{}/pcigale.ini'.format(pwd), 'r+') as file:
        data = file.readlines()
        data[11] = 'data_file = gal_pix_method1_{}.in'.format(str(redshift))
    
    with open('{}/pcigale.ini'.format(pwd), 'w') as file: 
        file.writelines(data)
    

if __name__ == '__main__':
    globals()[sys.argv[1]](float(sys.argv[2]), sys.argv[3], sys.argv[4])        #convert str input in terminal to float 
    
    
#Run CIGALE 


# -
