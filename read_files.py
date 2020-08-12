import glob
import os
from fnmatch import fnmatch
import csv
import scipy
import scipy.interpolate
import numpy as np
from astropy.io import ascii
import pandas as pd

class filters():
    """
        This class allows creation of objects for reading filters.

        Attributes:
            raw_filter_transmission (list): filter transmission coefficient
            raw_filter_wavelength (list): filter wavelength
            interpolated_transmission (obj): the interpolated filter transmission (numpy object)

        Methods:
            __init__: initializes the object
            getFilter: read filter file
            interpolateFilter: interpolate the filter
            getRawFilterWavelength: returns filter wavelength
            getRawFilterTransmission: returns filter transmission
            getInterpolatedFilter: returns interpolated filter transmission 
                (make sure to call interpolateFilter first or set auto_interpolate = True in __init__)
    """
    def __init__(self, filter_dir, filter_file ,auto_interpolate=False):
        """
            This method initializes the filter object. Will call method getFilter to read
            the filter file.

            Parameters:
                filter_dir (str): directory of the filter
                file (str): name of the filter file
                auto_interpolate (bool, optional): automatically  interpolate the filter using 
                    the default interpolation parameters (see method interpolateFilter).
                    Defaults to False.

            Returns:
                None

        """
        self.dir = filter_dir
        self.file = filter_file
        self.raw_filter_transmission = []
        self.raw_filter_wavelength = []
        self.getFilter()

        if auto_interpolate:
            self.interpolateFilter()

    def getFilter(self):
        """
            Read the filter file. Set raw_filter_transmission and 
            raw_filter_wavelength. Filter file wavelength must be in units of 1000 microns,
            transmission must be on a scale of 0-100.

            Parameter: None
            Returns: None

        """
        linecount = 0
        wavelength = []
        transmission = []
        with open(self.dir + self.file, 'r') as f:
            for line in f:
                linecount += 1 #ignore headerline
                if "#" not in line.strip() and len(line.strip()) > 0: #ignore header line
                    parts = line.strip().split("		") #how to split line
                    wavelength.append(float(parts[0])/1000) #convert to micron
                    transmission.append(float(parts[1])/100) #scale down to 0-1

                   
        #sort the list in ascending order
        wavelength, transmission =  zip(*sorted(zip(wavelength, transmission)))
        wavelength = list(wavelength)
        transmission = list(transmission)

        self.raw_filter_transmission = transmission
        self.raw_filter_wavelength =  wavelength

    def interpolateFilter(self, interpolation_method = scipy.interpolate.interp1d):
        """
            Interpolate the filter wavelength and transmission.

            Parameter: 
                interpolation_method (func, optional): how to interpolate. Defaults to scipy.interpolate.interp1d)

            Returns: None

        """
        self.interpolated_transmission = interpolation_method(x=self.getRawFilterWavelength(), y=self.getRawFilterTransmission())

    def getRawFilterWavelength(self):
        """
            Return the filter wavelength

            Parameter: None
            Returns: None

        """
        return self.raw_filter_wavelength

    def getRawFilterTransmission(self):
        """
            Return the filter transmission

            Parameter: None
            Returns: None

        """
        return self.raw_filter_transmission
    
    def getInterpolatedFilter(self):
        """
            Return the interpolated filter transmission

            Parameter: None
            Returns: None

        """
        return self.interpolated_transmission

class spectra():
    """
        This class allows the creation of objects for reading spectra. 

        There are three main important data structure in this class:
            wavelength (list): list of all wavelengths
            reflectance (list): list of all reflectance
            reflist (list): list of spectra name

        Data structure of wavelength, reflectance:
            [
                ['general category', [data1, data2 ],...],
                ['general category', [data1, data2 ],...],
                ...
            ]
            e.g.: 
            [
                ['ManMade',[wavelength1], [wavelength1], ... ],
                ['Minerals',[wavelength1, wavelength2],  ... ],
                ...
            ]
        Data structure of reference list:
            [
                ['general category', ['spectra name', 'spectra name'], ... ],
                ['general category', ['spectra name', 'spectra name'], ... ],
                ....
            ]
            e.g.:
            [
                ['ManMade',['Asphalt'], ['Asphalt_Tar'],      ... ],
                ['Minerals',['Acid_Mine_Dr', 'Acid_Mine_Dr'], ... ],
                ...
            ]
        
        This way, wavelength[i][j][k] corresponds to reflectance[i][j][k] 
                and reflist[i][j][k]

        Attributes:
            all_wavelength (list): wavelength list as described above
            all_reflectance (list): reflectance list as described above
            all_reflist (list): reference list as described above
        Methods:
            __init__: initializes the object
            readFiles: default file reading method
            getAllWavelengths: returns wavelength list
            getAllReflectance: returns refletance list
            getAllRefList: returns reference list
            setAllWavelengths: set wavelength list
            setAllReflectance: set refletance list
            setAllRefList: set reference list  
    """
    def __init__(self, dir, default_file_reading = True, 
                subdir_mode = False, file_reading_method = None,
                verbose = False):
        """
            This method initializes the object for reading spectra files. 

            

            Parameters:
                dir (str): directory and subdirectories to read.
                    e.g. dir = usgs_modified/*/*/ or dir = usgs_modified/ManMade/*/ or usgs_modified/ManMade/Asphalt/ 
                subdir_mode (bool, optional): If parameter dir have any subdirectories 
                    (usgs_modified/*/*/ or usgs_modified/ManMade/*/), please set subdir_mode to True.
                    Defaults to False.
                default_file_reading (bool, optional): True will use the default file reading method the comes with
                    the class (see method readFiles). Defaults to True.
                file_reading_method (func, optional): If default_file_reading is True, please set a function that
                    will read the files for the class. This function outputs HAVE to follow the data structure of 
                    the default file reading method. Function has to return wavelength, reflectance, reference list.
                    See the default data structure for more information.
                verbose (bool, optional): True allows the method to give updates as it runs.
                    Defaults to False.
            Returns:
                None
        """

        self.dir = dir
        self.all_wavelength = []
        self.all_reflectance = []
        self.all_reflist = []
        self.subdir_mode = subdir_mode
        self.verbose = verbose
        patterns = ["*.spectrum.txt", '*.csv', '*.asc']

        if default_file_reading == False:
            print("Using custom file reading method.")
            #self.all_wavelength, self.all_reflectance, self.all_ref_list = file_reading_method(dir)
        else:
            print("Starting to read spectra files with the default file reading Method")
            print("Reading directory: {}\nReading Subdirectory: {}".format(
                        self.dir, self.subdir_mode))
            
            #if no subdir, read as normal
            if subdir_mode == False:
                current_category  = self.dir.split("/")[1]
                
                if default_file_reading and file_reading_method == None:
                    self.all_wavelength, self.all_reflectance = self.readFiles(self.dir)
                    self.all_wavelength.insert(0,current_category)
                    self.all_reflectance.insert(0,current_category)
                    
                    files_to_read = [self.dir]
                    for pattern in patterns:
                        for _,_,files in os.walk(self.dir):
                            for f in files:
                                if fnmatch(f, pattern):
                                    files_to_read.append(f)
                    self.all_reflist = files_to_read
            #if subdir mode, make sure output is in desired structure
            else:
                
                wavelength = []
                reflectance = []
                ref_list = []
                current_category = ''
                for subdir in glob.glob(self.dir):
                    #get the category (e.g. ManMade)
                    category = subdir.split("/")[1]

                    if current_category != category:
                        current_category = category
                        ref_list.append([current_category])
                        wavelength.append([current_category])
                        reflectance.append([current_category])
                    #get wavelength and reflectance
                    current_wavelength, current_reflectance = self.readFiles(subdir)
                    #get name of the spectra (e.g. Asphalt)
                    current_type = [[subdir.split("/")[2]]]*len(current_wavelength)

                    #add to final list
                    wavelength[-1].append(current_wavelength)
                    reflectance[-1].append(current_reflectance)
                    ref_list[-1].append(current_type)
                #set to object list
                self.all_wavelength =  wavelength
                self.all_reflectance = reflectance
                self.all_reflist = ref_list
            print("Reading complete.")

    def getAllWavelengths(self):
        """
            Returns wavelength list. See data structure explanations in the class documentation.
        """
        return self.all_wavelength
    def getAllReflectance(self):
        """
            Returns reflectance list. See data structure explanations in the class documentation.
        """
        return self.all_reflectance
    def getAllRefList(self):
        """
            Returns reference list. See data structure explanations in the class documentation.
        """
        return self.all_reflist

    def setAllWavelengths(self, wavelength):
        """
            Set wavelength list. See data structure explanations in the class documentation.
        """
        self.all_wavelength = wavelength
    def setAllReflectance(self, reflectance):
        """
            Set reflectance list. See data structure explanations in the class documentation.
        """
        self.all_reflectance = reflectance
    def setAllRefList(self, reflist):
        """
            Set reference list. See data structure explanations in the class documentation.
        """
        self.all_reflist = reflist

    def readFiles(self, subdir):
        """
            Read all spectra files in lower subdirectories. Currently, will read formatted *.asc
            and *.csv files. Formated means: only data (no comments) in the file and one line of 
            header on top. First two columns of data must be wavelength, reflectance. 

            Parameter:
                subdir (str): directory to read files.
            Returns:
                wavelength (list): list of spectra
                reflectance (list): list of reflectance
        """
        wavelength = []
        reflectance = []

        #read USGS files. Make sure the data files are without the comments
        pattern = "*.asc"
        files_to_read = []
        for _,_,files in os.walk(subdir):
            for f in files:
                if fnmatch(f, pattern):
                    f = subdir + f
                    files_to_read.append(f)
                    
        if '.asc' in subdir: files_to_read = [subdir]
        for f in files_to_read:
            if self.verbose: print(f)
            with open(f, 'r') as f3:
                linecount = 0
                current_wavelength = []
                current_reflectance = []
                for line in f3:
                    linecount += 1
                    if linecount>1 and '*' not in line.strip().split(" ")[0]:
                        temp_data = []
                        for j in range(len(line.strip().split(" "))):
                            if len(line.strip().split(" ")[j]) > 0:
                                temp_data.append(line.strip().split(" ")[j])
                        
                        current_wavelength.append(float(temp_data[0]))
                        if '*' in temp_data[1]:
                            current_reflectance.append(float(temp_data[1].
                                                        replace("*","")))
                        else:
                            current_reflectance.append(float(temp_data[1]))                                    
                wavelength.append(current_wavelength)
                reflectance.append(current_reflectance)
        
        #read csv files (make sure data files are also without comments)
        pattern = "*.csv"
        files_to_read = []
        for _,_,files in os.walk(subdir):
            for f in files:
                if fnmatch(f, pattern):
                    f = subdir + f
                    files_to_read.append(f)

        if '.csv' in subdir: files_to_read = [subdir]
        for f in files_to_read:
            if self.verbose: print(f)
            with open(f, 'r') as f3:
                linecount = 0
                current_wavelength = []
                current_reflectance = []
                for line in f3:
                    linecount += 1
                    if linecount > 1:
                        current_wavelength.append(float(line.split(",")[0]))
                        current_reflectance.append(float(line.split(",")[1]))
                wavelength.append(current_wavelength)
                reflectance.append(current_reflectance)

        # #read ASTER/ECOSTRESS (files can be with comments)
        # pattern = "*.spectrum.txt"
        # files_to_read = []
        # for _,_,files in os.walk(subdir):
        #     for f in files:
        #         if fnmatch(f, pattern):
        #             f = subdir + f
        #             files_to_read.append(f)

        # for f in files_to_read:
        #     if self.verbose: print(f)
        #     data = open(f, 'r', encoding="utf8", errors='ignore').read()
        #     data = ascii.read(data,data_start=20).to_pandas()
        #     #print(data.values)
        #     wavelength.append(list(data[data.columns[0]]))
        #     reflectance.append(list(data[data.columns[1]]/100.))
        return wavelength, reflectance


class atmosphere_transmission():
    """
        Class to create objects to read atmosphere transmission. Atmosphere
        transmission MUST be a csv file.

        Attributes:
            transmission (list): transmission list
            wavelength (list): wavelength list
            reflectance (list): reflectance list

        Methods:
            __init__: initializes the object
            getInterpolation: interpolates the transmission
            getTransmission: returns the transmission list
            getWavelength: returns the wavelength list
        
    """
    def __init__(self,file, delim = ','):
        """
            This method initializes the object to read atmospher transmissions

            Parameters:
                file (str): path of file to read
                delim (str, optional): delimiter for csv reader. Defaults to ','
            Returns:
                None

        """
        reader = csv.reader(open(file,'r'), delimiter=delim)
        x = list(reader)
        result = np.array(x).astype('float')
        self.file = file
        self.transmission = result
        self.wavelength = result[:,0]
        self.reflectance = result[:,1]
    
    def getInterpolation(self, interpolation_scheme = scipy.interpolate.interp1d):
        """
            Interpolates the transmission. Returns the interpolated transmission.
        """
        interp = interpolation_scheme(self.wavelength, self.reflectance)
        self.interp = interp

        return self.interp
    def getTransmission(self):
        """
            Returns the transmission list.
        """
        return self.transmission
    def getWavelength(self):
        """
            Returns the wavelength list.
        """
        return self.wavelength
