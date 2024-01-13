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
import sys 
import numpy as np
import pandas as pd
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
from photutils.segmentation import make_source_mask
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
    
def precigale(input): 
    
    mergenew2 = pd.read_csv('mergenew2.csv')
    subset = mergenew2.head(100000)

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
    
    else: 
        run = subset.loc[subset['OBJID'] == input, 'RUN'].values[0]
        run6 = str(run).zfill(6)
        rerun = subset.loc[subset['OBJID'] == input, 'RERUN'].values[0]
        camcol = subset.loc[subset['OBJID'] == input, 'CAMCOL'].values[0]
        field = subset.loc[subset['OBJID'] == input, 'FIELD'].values[0]
        field4 = field4 = str(field).zfill(4)
        (ra,dec) = (subset.loc[subset['OBJID'] == input, 'RA'].values[0], subset.loc[subset['OBJID'] == input, 'DEC'].values[0])
        redshift = subset.loc[subset['OBJID'] == input, 'Z'].values[0]

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
    
    
    
    ###### Removing other sources (individual bands)
   
    # Clean function - individually cleans every band 
    def clean(data): 
        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (20, 20), filter_size=(3, 3),                        #changed filter_size from (2,2) to (3,3) bc needs odd size for both axes apparently, also changed box size to (30,30).. tweak parameters for cleaning 
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)         #look into (10,10) and (2,2) aka was originally (50,50) and (3,3)
        data -= bkg.background  # subtract the background
        threshold = 2. * bkg.background_rms  # above the background
    
        #convolving/smoothing, segmenting, and deblending 
        sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        convolved_data = convolve(data, kernel, normalize_kernel=True)
        npixels = 5
        segm = detect_sources(convolved_data, threshold, npixels=npixels)
        segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels,
                                       nlevels=32, contrast=0.1)
    
        #table of (selected) properties for each source
        cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
        columns = ['label', 'xcentroid', 'ycentroid', 'area', 'bbox', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin']
        tbl_data = cat.to_table(columns='data')
        tbl = cat.to_table(columns=columns)
        tbl['xcentroid'].info.format = '.4f'  # optional format
        tbl['ycentroid'].info.format = '.4f'
        #print(tbl)
        #print(tbl_data)

        #assume (101,101) is central galaxy to isolate central source
        galaxy = (tbl['xcentroid'] > 0.94*(data.shape[1]/2)) & (tbl['xcentroid'] < 1.06*(data.shape[1]/2)) & (tbl['ycentroid'] > 0.94*(data.shape[0]/2)) & (tbl['ycentroid'] < 1.06*(data.shape[0]/2))
        #print(tbl[galaxy])   
        n = tbl[galaxy][0]['label']         #get label/source number of central galaxy
    
        for k in tbl['label']:
            xmin = tbl[k-1]['bbox_xmin']
            xmax = tbl[k-1]['bbox_xmax']
            ymin = tbl[k-1]['bbox_ymin']
            ymax = tbl[k-1]['bbox_ymax']
            #replace ALL sources with 0 first
            data[ymin:ymax+1, xmin:xmax+1] = np.zeros((ymax+1 - ymin, xmax+1 - xmin), dtype=np.int64)

        #print(cutout.data) 
        #for bkg, select at random from any/all values that aren't = 0
        #needs to be a flattened 1d array
        bkg_cutout_flat = data.ravel()  
        bkg_cutout_filtered = bkg_cutout_flat[np.nonzero(bkg_cutout_flat)]        #everything that is not 0's

        #another loop to replace source_cutouts with background stuff (b/c has to iterate through all k's to replace all of those w 0)
        for k in tbl['label']:
            xmin = tbl[k-1]['bbox_xmin']
            xmax = tbl[k-1]['bbox_xmax']
            ymin = tbl[k-1]['bbox_ymin']
            ymax = tbl[k-1]['bbox_ymax']
        
            if k == n:   #replace central galaxy (set to 0) with original data
                data[ymin:ymax+1, xmin:xmax+1] = tbl_data[k-1]['data']           
            
            else:        #replace other sources (now with 0's) with sky pixel values 
                data[ymin:ymax+1, xmin:xmax+1] = np.random.choice(bkg_cutout_filtered, size=(ymax+1 - ymin, xmax+1 - xmin))

        data += bkg.background           #add background back
        #plt.imshow(data, origin='lower')
    
        return data
    

    # pixel fluxes for cleaned data
    clean_u = clean(cutout_u.data)
    clean_g = clean(cutout_g.data)
    clean_r = clean(cutout_r.data)
    clean_i = clean(cutout_i.data)
    clean_z = clean(cutout_z.data)

    
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


    ###### Filter Observation Map ######

    ## The filter observation map is a 2d array with the same dimension as the images.
    FILTER_NAMES  = ['u', 'g', 'r', 'i', 'z']

    nCols = 1
    nRows = 5

    xSize = 15
    ySize = xSize*float(nRows)/float(nCols)

    wspace = 0.0
    hspace = 0.0

    #def_plot_values_extra_large()
    fig = plt.figure(figsize=(xSize, ySize))
    gs = gridspec.GridSpec(nrows=nRows, ncols=nCols, wspace=wspace, hspace=hspace)

    ########

    filterObsMap = np.zeros_like(images[0])
    
    #Plot sources (of cleaned images)
    for i in range(nRows):
        for j in range(nCols):
            k  = i*nCols + j
        
            if (k < 5):
                ax = plt.subplot(gs[k])
        
                text = FILTER_NAMES[k]+" Borders"
            
                ax.text(0.03, 0.96, text, color='white', ha='left', va='top', transform=plt.gca().transAxes)
            
                data = deepcopy(images[k])

                sigma_clip = SigmaClip(sigma=3.)
                bkg_estimator = MedianBackground()
                bkg = Background2D(data, (20, 20), filter_size=(3, 3),
                                sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)         #look into (10,10) and (2,2) aka was originally (50,50) and (3,3)
                data -= bkg.background  # subtract the background
                threshold = 2. * bkg.background_rms  # above the background

                #convolving/smoothing, segmenting, and deblending 
                sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
                kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
                convolved_data = convolve(data, kernel, normalize_kernel=True)
                npixels = 5
                segm = detect_sources(convolved_data, threshold, npixels=npixels)
                segm_deblend = deblend_sources(convolved_data, segm, npixels=npixels, nlevels=32, contrast=0.1)
 
                #table of (selected) properties for each source
                cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
                columns = ['label', 'xcentroid', 'ycentroid', 'area', 'bbox', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin']
                tbl_data = cat.to_table(columns='data')
                tbl = cat.to_table(columns=columns)
                tbl['xcentroid'].info.format = '.4f'  # optional format
                tbl['ycentroid'].info.format = '.4f'
    
                data += bkg.background           #add background back

                print(FILTER_NAMES[k], segm.areas, tbl)   # segm.background_area,
            
                outline = segm.outline_segments()            
            
                data[outline > 0] = np.inf
                                    
                ## ax.imshow(masked, cmap = 'Reds')
            
                #ax.imshow(data, origin='lower', aspect='equal', vmin=vmin, vmax=vmax, interpolation='nearest', cmap='gray')
            
                ax.imshow(data, origin='lower', aspect='equal', interpolation='nearest', cmap='gray')       #comment out if don't want to see 

                filterObsMap[segm.data > 0] += 1.0
            
                if (i < (nRows-1)):
                    ax.set_xticklabels('')
                if (j > 0):
                    ax.set_yticklabels('')

    xLabel, yLabel = r'$X$ [pix]', r'$Y$ [pix]'

    #axCommons = drawCommonLabel(xLabel, yLabel, fig, xPad=20, yPad=25)

    
    # Plot filter observation map
    xSize = 10
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
    
    
    # Nonzero indices
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

    gal_indices.to_csv('gal_indices', index=False)
    
    

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
    
    gal = ascii.write(gal, 'gal_pix_method1.in'.format(input), overwrite=True)
    
    print(gal) 

    
if __name__ == '__main__':
    globals()[sys.argv[1]](float(sys.argv[2]))        #convert str input in terminal to float 
    
    
#Run CIGALE 


# -
