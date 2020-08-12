
import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.stats
import read_files
import pandas as pd
from sklearn.metrics import confusion_matrix

def spectra_to_color(wavelength, reflectance, 
    filter, interpolated_filter,
    resolution=1000, interpolation_scheme = scipy.interpolate.interp1d, 
    integration_scheme = scipy.integrate.trapz):

    """ 
        This function takes in a spectra and a filter and returns the color.

        Parameters:
            wavelength (list): 1D list of wavelength
            reflectance (list): 1D list of reflectance
            filter (list): 2D filter list in the form 
                            [[wavelength], [transmission]]
            interpolated_filter (obj): scipy interpolated object of filter
            resolution (int, optional): how sparse the interpolation. 
                            Defaults to 1000
            interpolation_scheme (func, optional): function used to interpolate.
                            Defaults to scipy.interpolate.interp1d
            integration_scheme (func, optional): function used to integrate.
                            Defaults to scipy.integrate.trapz
        Returns:
            (int) : the color
    """
    #interpolate the spectra
    interp_reflectance = interpolation_scheme(wavelength, reflectance)
    #get the bounds for the grid to multiply filter and spectra
    lower = max(min(wavelength),min(filter[0]))
    upper = min(max(wavelength),max(filter[0]))
    grid = np.linspace(lower,upper, num=resolution, endpoint=True)
    #get the interpolated transmission
    transmission = interpolated_filter
    #multiply the filter and spectra
    product = interp_reflectance(grid) * transmission(grid)
    #integrate over the product to get the magnitude of the color
    magnitude = integration_scheme(y = product,x = grid)

    return -2.5*np.log10(magnitude)

def spectra_to_flux(wavelength, reflectance, 
    filter, interpolated_filter,
    resolution=1000, interpolation_scheme = scipy.interpolate.interp1d, 
    integration_scheme = scipy.integrate.trapz):

    """ 
        This function takes in a spectra and a filter and returns the convolved flux.

        Parameters:
            wavelength (list): 1D list of wavelength
            reflectance (list): 1D list of reflectance
            filter (list): 2D filter list in the form 
                            [[wavelength], [transmission]]
            interpolated_filter (obj): scipy interpolated object of filter
            resolution (int, optional): how sparse the interpolation. 
                            Defaults to 1000
            interpolation_scheme (func, optional): function used to interpolate.
                            Defaults to scipy.interpolate.interp1d
            integration_scheme (func, optional): function used to integrate.
                            Defaults to scipy.integrate.trapz
        Returns:
            (int) : the color difference in filter 1 and filter 2
    """
    #interpolate the spectra
    interp_reflectance = interpolation_scheme(wavelength, reflectance)
    #get the bounds for the grid to multiply filter and spectra
    lower = max(min(wavelength),min(filter[0]))
    upper = min(max(wavelength),max(filter[0]))
    grid = np.linspace(lower,upper, num=resolution, endpoint=True)
    #get the interpolated transmission
    transmission = interpolated_filter
    #multiply the filter and spectra
    product = interp_reflectance(grid) * transmission(grid)
    #integrate over the product to get the magnitude of the color
    magnitude = integration_scheme(y = product,x = grid)

    return magnitude

def all_spectra_to_flux(dataframe_dict, filter_profile, filter_interp,
                         error_report = True):
    """
        This function takes in a dict of DataFrame of spectra - one 
        DataFrame of spectra is one category (e.g. ManMade), - to calculate 
        the color in a filter profile.

        Parameters:
            dataframe_dict (dict): dict of DataFrame is what function 
                                        sql_to_dataframe returns 

            filter_profile (list): 2D filter list in the form 
                            [[wavelength], [transmission]] for filter 
            filter_interp (obj): numpy interpolated object of filter

            error_report (bool, optional): If True will notify of all 
                Interpolation Error. Defaults to True
        Returns:
            (list), (list): returns two lists of color differences for all
                spectra. Each list have structure:
                e.g. 
                ["
                    ['B' 'B' 'B' 'B']",
                    [
                        ['ManMade', colordiff1, ... ],
                        ['Minerals', colordiff1, ...],
                        ...
                    ]
                ]
    """
    color = []
    error = 0
    #loop through dict to get each DataFrame (ManMade, Minerals, etc.)
    for key in dataframe_dict.keys():
        #get each DataFrame
        df = dataframe_dict[key]
        #set the "category". e.g. ManMade, Minerals, etc.
        color_temp = [key] 
        for i in range(len(df)):
            #get the wavelength and reflectance from the DataFrame
            wavelength = df.iloc[i,1]
            reflectance = df.iloc[i,2]
            #try-except in case of interpolation error
            try:
                #calculate the color
                c = spectra_to_flux(wavelength, reflectance,
                filter_profile, filter_interp)
                #add to temporary list
                color_temp.append(c)
            except ValueError:
                #report error if error_report == True
                if error_report:
                    print("Interpolation Error.")
                    error += 1
                else:
                    #do nothing otherwise
                    pass
        #add to final result list
        color.append(color_temp)
    return color

def spectra_to_color_difference(wavelength, reflectance, 
    filter1, interpolated_filter1, filter2, interpolated_filter2, 
    resolution=1000, interpolation_scheme = scipy.interpolate.interp1d , 
    integration_scheme = scipy.integrate.trapz):

    """ 
        This function takes in a spectra and two filters and give the 
        color difference.

        Parameters:
            wavelength (list): 1D list of wavelength
            reflectance (list): 1D list of reflectance
            filter1 (list): 2D filter list in the form 
                            [[wavelength], [transmission]]
            interpolated_filter1 (obj): scipy interpolated object of filter 1
            filter2 (list): 2D filter list in the form 
                            [[wavelength], [transmission]]
            interpolated_filter2 (obj): scipy interpolated object of filter 2

            resolution (int, optional): how sparse the interpolation. 
                            Defaults to 1000
            interpolation_scheme (func, optional): function used to interpolate.
                            Defaults to scipy.interpolate.interp1d
            integration_scheme (func, optional): function used to integrate.
                            Defaults to scipy.integrate.trapz
        Returns:
            (int) : the color difference in filter 1 and filter 2
    """
    #interpolate the spectra
    interp_reflectance = interpolation_scheme(wavelength, reflectance)
    #get the bounds for the grid to multiply filter and spectra
    lower_c1 = max(min(wavelength),min(filter1[0]))
    upper_c1 = min(max(wavelength),max(filter1[0]))
    grid_c1 = np.linspace(lower_c1,upper_c1, num=resolution, endpoint=True)
    #get the interpolated transmission
    c1_transmission = interpolated_filter1
    #multiply the filter and spectra
    c1_product = interp_reflectance(grid_c1) * c1_transmission(grid_c1)
    #integrate over the product to get the magnitude of the color
    c1_magnitude = integration_scheme(y = c1_product,x = grid_c1)

    lower_c2 = max(min(wavelength),min(filter2[0]))
    upper_c2 = min(max(wavelength),max(filter2[0]))
    grid_c2= np.linspace(lower_c2,upper_c2, num=resolution, endpoint=True)
    c2_transmission = interpolated_filter2
    c2_product = interp_reflectance(grid_c2) * c2_transmission(grid_c2)
    c2_magnitude = integration_scheme(y = c2_product,x = grid_c2)

    #return the color difference
    return -2.5*np.log10(c1_magnitude/c2_magnitude)

def all_spectra_to_color(dataframe_dict, filter_profile, filter_interp,
                         error_report = True):
    """
        This function takes in a dict of DataFrame of spectra - one 
        DataFrame of spectra is one category (e.g. ManMade), - to calculate 
        the color in a filter profile.

        Parameters:
            dataframe_dict (dict): dict of DataFrame is what function 
                                        sql_to_dataframe returns 

            filter_profile (list): 2D filter list in the form 
                            [[wavelength], [transmission]] for filter 
            filter_interp (obj): numpy interpolated object of filter

            error_report (bool, optional): If True will notify of all 
                Interpolation Error. Defaults to True
        Returns:
            (list), (list): returns two lists of color differences for all
                spectra. Each list have structure:
                e.g. 
                ["
                    ['B' 'B' 'B' 'B']",
                    [
                        ['ManMade', colordiff1, ... ],
                        ['Minerals', colordiff1, ...],
                        ...
                    ]
                ]
    """
    color = []
    error = 0
    #loop through dict to get each DataFrame (ManMade, Minerals, etc.)
    for key in dataframe_dict.keys():
        #get each DataFrame
        df = dataframe_dict[key]
        #set the "category". e.g. ManMade, Minerals, etc.
        color_temp = [key] 
        for i in range(len(df)):
            #get the wavelength and reflectance from the DataFrame
            wavelength = df.iloc[i,1]
            reflectance = df.iloc[i,2]
            #try-except in case of interpolation error
            try:
                #calculate the color
                c = spectra_to_color(wavelength, reflectance,
                filter_profile, filter_interp)
                #add to temporary list
                color_temp.append(c)
            except ValueError:
                #report error if error_report == True
                if error_report:
                    print("Interpolation Error.")
                    error += 1
                else:
                    #do nothing otherwise
                    pass
        #add to final result list
        color.append(color_temp)
    return color

def all_spectra_to_color_difference(dataframe_dict,
    color1_profile, color2_profile, color3_profile, color4_profile,
    color1_interp, color2_interp, color3_interp, color4_interp,
    error_report = True):
    """
        This function takes in a dict of DataFrame of spectra - one 
        DataFrame of spectra is one category (e.g. ManMade), - to calculate 
        the color difference in filters 1 and 2 for all spectra sample.

        Parameters:
            dataframe_dict (dict): dict of DataFrame is what function 
                                        sql_to_dataframe returns 

            color1_profile (list): 2D filter list in the form 
                            [[wavelength], [transmission]] for filter 1
            color2_profile (list): 2D filter list in the form 
                            [[wavelength], [transmission]] for filter 2
            color3_profile (list): 2D filter list in the form 
                            [[wavelength], [transmission]] for filter 3
            color4_profile (list): 2D filter list in the form 
                            [[wavelength], [transmission]] for filter 4

            color1_interp (obj): numpy interpolated object of filter 1
            color2_interp (obj): numpy interpolated object of filter 2
            color3_interp (obj): numpy interpolated object of filter 3
            color4_interp (obj): numpy interpolated object of filter 4

            error_report (bool, optional): If True will notify of all 
                Interpolation Error. Defaults to True
        Returns:
            (list), (list): returns two lists of color differences for all
                spectra. Each list have structure:
                e.g. 
                ["
                    ['B' 'B' 'B' 'B']",
                    [
                        ['ManMade', colordiff1, ... ],
                        ['Minerals', colordiff1, ...],
                        ...
                    ]
                ]
    """
    colordiff1, colordiff2 = [], []
    error = 0
    #loop through dict to get each DataFrame (ManMade, Minerals, etc.)
    for key in dataframe_dict.keys():
        #get each DataFrame
        df = dataframe_dict[key]
        #set the "category". e.g. ManMade, Minerals, etc.
        category_c1c2, category_c3c4 = [key], [key] 
        for i in range(len(df)):
            #get the wavelength and reflectance from the DataFrame
            wavelength = df.iloc[i,1]
            reflectance = df.iloc[i,2]
            #try-except in case of interpolation error
            try:
                #calculate the color difference for filter 1 and filter 2
                c1_c2 = spectra_to_color_difference(wavelength, reflectance,
                color1_profile, color1_interp, color2_profile , color2_interp)
                #calculate the color difference for filter 3 and filter 4
                c3_c4 = spectra_to_color_difference(wavelength, reflectance,
                color3_profile, color3_interp, color4_profile , color4_interp)
                #add to temporary list
                category_c1c2.append(c1_c2)
                category_c3c4.append(c3_c4)
            except ValueError:
                #report error if error_report == True
                if error_report:
                    print("Interpolation Error.")
                    error += 1
                else:
                    #do nothing otherwise
                    pass
        #add to final result list
        colordiff1.append(category_c1c2)
        colordiff2.append(category_c3c4)
    return colordiff1, colordiff2

def dist(x1,y1,x2,y2):
    """
        Returns the Cartesian distance of two points

        Parameters:
            x1, y1, x2, y2 (float): coordinates of two points
        Returns:
            (int): distance between two points
    """
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def all_colorspace_distance(colordiff1,colordiff2, organism):
    """
        This function calculates the mean average distance of organic
        to inorganic points on the color-color space between two
        color difference (e.g. B-I vs I-V). 

        Parameters:
            colordiff1, colordiff2 (list): list structure is what function
                all_spectra_to_color_difference returns
            organism (list): list of labels of categories which are organic.
                e.g. ['VegetationandMicroorganism',...]
        Returns:
            (int): the mean average separation
    """
    #creates list for organism and non-organisms. [colordiff1, colordiff2]
    organisms_list = [[],[]]
    non_organisms_list = [[],[]]

    #sort colordiff1 and colordiff2 into organic and inorganic
    for i in range(len(colordiff1)):
        if colordiff1[i][0] in organism:
            organisms_list[0] += colordiff1[i][1:]
            organisms_list[1] += colordiff2[i][1:]
        else:
            non_organisms_list[0] += colordiff1[i][1:]
            non_organisms_list[1] += colordiff2[i][1:]

    #make it into np.array
    non_organisms_list = np.array(non_organisms_list)
    organisms_list = np.array(organisms_list)


    separation = []
    mean_separation = []
    #loop through each organism and calculate their distance to inorganic
    for i in range(len(organisms_list[0])):
        #calculate the distance vector
        distance = dist(organisms_list[0][i],organisms_list[1][i],
                        non_organisms_list[0],non_organisms_list[1])
        #store result
        separation.append(distance)
        #store the average distance
        mean_separation.append(np.mean(distance))

    #option graphing
    #print("average mean separation", np.mean(mean_separation))
    #print(scipy.stats.describe(mean_separation))
    #plt.hist(mean_separation, bins=50)
    #plt.show()

    #returns the mean average distance
    return np.mean(mean_separation)

def getInterpolation(wavelength, reflectance, 
                    interpolation_scheme = scipy.interpolate.interp1d):

    """
        This function returns the interpolate object from scipy of a given
        wavelength and reflectance.

        Parameters:
            wavelength (list): 1D list of wavelength data
            reflectance (list): 1D list of reflectance data
            interpolate_scheme (func, optional): the function used to
                interpolate. Defaults to scipy.interpolate.interp1d
        Returns:
            (obj): the scipy interpolate object
    """
        
    interp = interpolation_scheme(wavelength, reflectance)
    return interp

def multiply_spectra(interpolated1, interpolated2,
    spectra1_wavelength, spectra2_wavelength,
    interpolation_scheme = scipy.interpolate.interp1d,
    resolution = 1000, xlim = None):   
        
    """
        Multiply two spectra together using their resepective 
        interpolate objects.

        Parameters:
            interpolated1 (obj): scipy interpolate object of 
                    spectra 1
            interpolated2 (obj): scipy interpolate object of 
                    spectra 2
            spectra1_wavelength (list): list of spectra 1 wavelength
            spectra2_wavelength (list): list of spectra 2 wavelength
            interpolation_scheme (func, optional): function used to
                interpolate. Defaults to scipy.interpolate.interp1d
            resolution (int, optional): how sparse the interpolation resolution.
                Defaults to 1000
            xlim (list, optional): the lower and upper limit to multiply. If None,
                the function will find the lowest and highest possible limit.
                Defaults to None. 
        Returns:
            (list), (list): the grid and product of the two interpolated spectra

    """
    if xlim == None:
        #get the bounds for grid then make the grid
        lower = max(min(spectra1_wavelength),min(spectra2_wavelength))
        upper = min(max(spectra1_wavelength),max(spectra2_wavelength))
    else:
        lower = xlim[0]
        upper = xlim[1]
            
    grid = np.linspace(lower,upper, num=resolution, endpoint=True)

    #multiply the two spectra together
    product = interpolated1(grid) * interpolated2(grid)

    return(grid, product)

def rayleigh_scattering(lower, upper, earth_sun_toa_interp, resolution = 1000,
                        p = 1.0):
    """
        Returns the additional term correction for Rayleigh scattering.
        Taken from Allen's Astrophysical Quantities (2000).

        Parameter:
            lower (float): lower wavelength bound (in micron)
            upper (float): upper wavelength bound (in micron)
            earth_sun_toa_interp (obj): scipy object, interpolated spectra of 
                Earth top of atmosphere flux.
            resolution (int, optional): resolution of the Rayleigh scattering.
                Defaults to 1000
            p (float, optional): Pressure ratio. Defaults to 1.
         Returns:
            (numpy.array): the Rayleigh scattering correction
    """
    rayleigh_x = np.linspace(lower,upper,num=resolution)
    rayleigh_tau = 0.008569 * rayleigh_x**(-4) *(1 + 0.0113 * rayleigh_x**(-2) + 0.00013 * rayleigh_x**(-4))
    rayleigh_tau = rayleigh_tau*p
    rayleigh_scattering = 1-np.exp(-rayleigh_tau)

    rayleigh_interp = getInterpolation(rayleigh_x, rayleigh_scattering)
    rayleigh = rayleigh_interp(rayleigh_x) * earth_sun_toa_interp(rayleigh_x)
    return np.array(rayleigh)

def custom_spectra(composition_guide,xlimit,atmosphere_transmission_path, 
                    top_of_atmosphere_transmission, rayleigh = True, output='spectra', color_profiles=None):
    """
        Create a custom spectra (and can calculate color if provided a color profile) given a composition.

        Parameters:
            composition_guide (dict): must in the form
                {'composition name': [composition percentage, wavelength list, reflectance list],
                 ... etc ...}
            xlimit (list): in the form of [lower limit, upper limit]
            atmosphere_transmission_path (str): string of the path to the atmosphere transmission file
            top_of_atmosphere_transmission (str): string of the path to the top of atmosphere 
                                    transmission file
            rayleigh (bool, optional): True or False. 
                                    If True, adds on the Rayleigh scattering. Default is True.
            output (str): options are 'spectra', 'color', or 'both'.
            color_profile (list, required if output is 'color' or 'both'): in the form
                                    [[color_filter_profile, color_interpolated],etc...]
        Returns:
            if 'spectra': returns a pandas dataframe that contains the wavelength and reflectance
            if 'color': returns the color values list
            if 'both': returns dataframe, color list
    """

    #make sure that color profiles is not empty
    if (output=='color' or output=='both') and color_profiles == None:
        raise ValueError

    #get atmos transmission
    transmission = read_files.atmosphere_transmission(atmosphere_transmission_path)
    transmission_interp = transmission.getInterpolation()
    
    #multiply the transmission by composition component 
    composition_interpolated = []
    for name, data in composition_guide.items():
        wavelength = data[1]
        reflectance = data[2]
        composition_percentage = data[0]

        composition_interp = getInterpolation(wavelength,(composition_percentage/100.)*np.array(reflectance))
        composition_interpolated.append(multiply_spectra(transmission_interp, composition_interp,
                                    transmission.getWavelength(), wavelength,
                                    xlim = xlimit))

    #add different components together                                    
    final_spectra_x, final_spectra_y = [], []
    for composition in composition_interpolated:
        composition_wavelength = composition[0]
        composition_reflectance = composition[1]

        if len(final_spectra_y) > 0:
            final_spectra_y = np.array(final_spectra_y) + np.array(composition_reflectance)
        else:  
            final_spectra_x = composition_wavelength
            final_spectra_y = composition_reflectance

    #add on rayleigh scattering if necessary
    if rayleigh:
        toa = read_files.atmosphere_transmission(top_of_atmosphere_transmission)
        toa_interp = toa.getInterpolation()

        rayleigh_array = rayleigh_scattering(xlimit[0],xlimit[1],toa_interp)
        final_spectra_y = np.array(final_spectra_y) + np.array(rayleigh_array)

    #make it a pandas dataframe
    final_spectra_y = pd.DataFrame({'empty':[0],'wavelength':[final_spectra_x],
                                            'reflectance':[final_spectra_y]})

    #output
    if output == 'spectra':
        return final_spectra_y

    elif output == 'color':
        color_list = []
        #calculate in each color
        for color in color_profiles:
            spectra_color = all_spectra_to_color({'spectra':final_spectra_y}, 
                                color[0], color[1], error_report=False)
            spectra_color = np.array(spectra_color[0][1:])
            color_list.append(spectra_color)

        return color_list
    elif output=='both':
        color_list = []
        for color in color_profiles:
            spectra_color = all_spectra_to_color({'spectra':final_spectra_y}, 
                                color[0], color[1], error_report=False)
            spectra_color = np.array(spectra_color[0][1:])
            color_list.append(spectra_color)

        #return both dataframe and color lists
        return final_spectra_y, color_list
    
def add_noise(snr, signal):
    """ 
        This method returns random Gaussian noise to the data, given the SNR

    """
    
    if type(snr) == None:
        return 0
    
    noise = []
    for i in range(len(signal)):
        #get the signal
        s = signal[i]

        #get the snr if input snr is a distribution
        if type(snr) not in [int, float]:
            signal_noise_ratio = snr[i]
        else:
            signal_noise_ratio = snr
        
        #generate random noise, make sure that we don't end up with negative flux
        counter = 0
        while counter < 1000:
            tmp = np.random.normal(loc=0.0,scale=s/signal_noise_ratio)
            #make sure no negative flux!
            if tmp + s > 0:
                noise.append(tmp)
                break
            counter += 1
        #if counter >0 : print(s,counter)

    return np.array(noise)

def tnr(y_true, y_pred):
    """
        This method calculates the True Negative Rate
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr_val = tn/(tn+fp)
    return tnr_val    