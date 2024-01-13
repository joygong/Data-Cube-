# All galaxy pixels; spectra plots 

#id = gal_indices.loc[gal_indices['y_data_index'] == i, 'id'].values[0]
#id
#gal_indices.iloc[250:]


# get id for the central pixels + outskirts 
gal_indices
#center = gal_indices[(gal_indices['y_data_index'] >= 95) & (gal_indices['y_data_index'] <= 105) & (gal_indices['x_data_index'] >= 95) & (gal_indices['x_data_index'] <= 105)]

#for i in center['id']:
for i in gal_indices['id']:
    def name(i): 
        name = '{}_best_model.fits'.format(i)
        return name  

    file = fits.open(name(i))
    fnu = file[1].data['Fnu']
    wavelength = file[1].data['wavelength']
    # Convert from Fnu to Flambda
    flambda = 1e-29 * 1e+9 * fnu / (wavelength * wavelength) * c
    
    #y = center.loc[center['id'] == i, 'y_data_index'].values[0]
    #x = center.loc[center['id'] == i, 'x_data_index'].values[0]
    y = gal_indices.loc[gal_indices['id'] == i, 'y_data_index'].values[0]
    x = gal_indices.loc[gal_indices['id'] == i, 'x_data_index'].values[0]
    
    # plot spectra (flux vs. wavelength)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Wavelength (nanometer)')
    plt.ylabel('Flambda (W/m²/nm)')
    plt.title('Pixel ({},{})'.format(x,y))
    plt.plot(wavelength, flambda)
    plt.show()
    #print(x,y)
