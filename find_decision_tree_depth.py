from matplotlib.pyplot import *
from numpy import *

import pandas as pd
import pickle as pk
from matplotlib import cm

#for the accuracy score
from sklearn.metrics import balanced_accuracy_score, accuracy_score

#for scaling
from sklearn import preprocessing

#for splitting dataset
from sklearn.model_selection import train_test_split

#the ML models we are using (single methods)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
#ensemble method
from sklearn.ensemble import VotingClassifier

from scipy import optimize

#methods available to read spectra data files
import read_files
#important helper methods
import main_methods
#import standard BVRI filters
import bvri_filters

#for multiprocessing
from joblib import Parallel, delayed
import gc

#miscellaneous important things
import time
from datetime import datetime
import warnings
import sys
from copy import copy, deepcopy
from decimal import Decimal
import argparse

special_read_list = {'agrococcus':'spectra/Hegde_2013/Agrococcus/Agrococcus%20sp._KM349956.csv',
                     'geodermatophilus': 'spectra/Hegde_2013/Geodermatophilus/Geodermatophilus%20sp._KM349882.csv',
                     'bark': 'spectra/ECOSTRESS_nonphotosyntheticvegetation/bark/nonphotosyntheticvegetation.bark.acer.rubrum.vswir.acru-1-81.ucsb.asd.spectrum.csv',
                     'lichen': 'spectra/ECOSTRESS_nonphotosyntheticvegetation/lichen/nonphotosyntheticvegetation.lichen.lichen.species.vswir.vh298.ucsb.asd.spectrum.csv',
                     'aspen': 'spectra/USGS_Vegetation_and_Microorganisms/Aspen_Leaf-A/aspen_leaf_dw92-2.10753.asc',
                     'leafy spurge': 'spectra/USGS_Vegetation_and_Microorganisms/LeafySpurge/leafyspurge_spurge-a2-jun98.11306.asc'
                    }
MAX_ITER = 100
DEFAULT_INPUT_ML_DATA_PATH = 'data/ML_data_cloud_rayleigh/ML_'
all_model_names = ['LR','LDA','KNN','CART','NB','SVM','MVH','MVS']


################################# PARAMETERS ##################################

parser = argparse.ArgumentParser()
parser.add_argument('-mpc', '--cores',  type=int, default=10, required=True,
                    help='How many cores for multiprocessing?' )

parser.add_argument('-ob', '--binaryoutput', type=str,
                    default = 'output/binary_class',
                    help='Binary output directory' )
parser.add_argument('-om', '--multioutput', type=str,
                    default = 'output/multiclass',
                    help='Multiclass output directory' )
parser.add_argument('-smin', '--snrmin', type=int, default=1,
                    help='Minimum SNR to train' )
parser.add_argument('-smax', '---snrmax',  type=int, default=100,
                    help='Maximum SNR to train' )
parser.add_argument('-s', '--seed', type=int, default=10,
                help='Seed for the random state. Set to None for full random' )

args = parser.parse_args()

SNR_max = args.snrmax                     
SNR_min = args.snrmin                       
cores = args.cores                       
seed = args.seed                       
binary_output_dir = args.binaryoutput
multiclass_output_dir = args.multioutput

print('Training on all vegetations')
print('Output directories: {}, {}'.format(binary_output_dir, 
                                            multiclass_output_dir))
                    
time_of_run = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
initial_time = time.time()
np.random.seed(seed)

print("Using {} cores".format(cores))
print("Seed: {}".format(seed))
print("SNR: [{}, {}]".format(SNR_min, SNR_max))
###############################################################################

### HELPER FUNCTIONS FOR MULTIPROCESSING ###
def make_training_data(category_name, SNR_min, SNR_max):
    df_training_spectra = pd.read_pickle(DEFAULT_INPUT_ML_DATA_PATH+category_name.replace(' ','')+'.pkl')

    print('Creating noisy data for {} (t={})'.format(category_name, time.time() - initial_time))
    #make color from spectra
    df_training_i = main_methods.all_spectra_to_flux( 
        {'training':df_training_spectra}, bvri_filters.i_filter_profile, bvri_filters.i_interpolated, error_report=False)

    df_training_b = main_methods.all_spectra_to_flux( 
        {'training':df_training_spectra}, bvri_filters.b_filter_profile, bvri_filters.b_interpolated, error_report=False)

    df_training_v = main_methods.all_spectra_to_flux( 
        {'training':df_training_spectra}, bvri_filters.v_filter_profile, bvri_filters.v_interpolated, error_report=False)

    df_training_r = main_methods.all_spectra_to_flux( 
        {'training':df_training_spectra}, bvri_filters.r_filter_profile, bvri_filters.r_interpolated, error_report=False)

    df_training_b_0 = np.array(df_training_b[0][1:])
    df_training_v_0 = np.array(df_training_v[0][1:])
    df_training_r_0 = np.array(df_training_r[0][1:])
    df_training_i_0 = np.array(df_training_i[0][1:])
    
    snr = np.random.uniform(low=SNR_min, high=SNR_max, 
                            size=len(df_training_b_0))

    df_training_b = df_training_b_0 + main_methods.add_noise(snr, df_training_b_0)
    df_training_v = df_training_v_0 + main_methods.add_noise(snr, df_training_v_0)
    df_training_r = df_training_r_0 + main_methods.add_noise(snr, df_training_r_0)
    df_training_i = df_training_i_0 + main_methods.add_noise(snr, df_training_i_0)
    
    df_training_spectra['B'] = df_training_b
    df_training_spectra['V'] = df_training_v
    df_training_spectra['R'] = df_training_r
    df_training_spectra['I'] = df_training_i


    #make classifier for supervised training
    training_class = []
    for i in df_training_spectra['classification']:
        if i['water'] > 0:
            training_class.append(1) #true, vegetation
        elif i['water'] == 0:
            training_class.append(0) #false, no vegetation
        else:
            raise ValueError

    training = []
    for i in range(len(df_training_i)):
        training.append([df_training_b[i],df_training_v[i],df_training_r[i],df_training_i[i]])

    training_class = array(training_class)
    training = array(training)

    #vegetation %
    validation_veg_comp = []

    for i in range(len(df_training_spectra['B'])):
        veg = df_training_spectra.iloc[i]['classification']['tree']
        validation_veg_comp.append(veg)
    
    gc.collect()
    return training, training_class, validation_veg_comp, category_name

def DecisionTree_find_alpha_helper(ccp_alpha,X_train, y_train):

    return DecisionTreeClassifier(random_state=seed, ccp_alpha=ccp_alpha).fit(
            X_train, y_train)

def get_alpha(category_name, training, training_class):
    X_train, X_test, y_train, y_test = train_test_split(training, 
                                            training_class, random_state=seed)
    
    clf = DecisionTreeClassifier(random_state=seed)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    clfs = []

    select_list = list(arange(0, len(ccp_alphas), 100))
    select_list = unique(array(select_list))
    ccp_alphas, impurities = ccp_alphas[select_list], impurities[select_list]

    clfs = Parallel(n_jobs=cores, verbose=10)(delayed(
        DecisionTree_find_alpha_helper)(ccp_alpha,X_train, y_train)
        for ccp_alpha in ccp_alphas
    )

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    index_best_alpha = test_scores.index(max(test_scores))
    index_min_bound  = max(0,index_best_alpha - 1)
    index_max_bound  = min(len(ccp_alphas)-1, index_best_alpha + 1)

    res = optimize.minimize_scalar(fun=opt_func, method='brent',
        bracket = (ccp_alphas[index_min_bound], ccp_alphas[index_max_bound]),
        args=(X_train, y_train, X_test, y_test), 
        options = {'maxiter': MAX_ITER})

    print(res)

    return res.x

def opt_func(x, X_train, y_train, X_test, y_test):
    
    """
        Objective function to minimize
    """
    
    #ccp_alpha must be > 0
    if x < 0:
        return np.inf
    
    clf = DecisionTreeClassifier(random_state=10, ccp_alpha=x)
    clf.fit(X_train, y_train)
    return -clf.score(X_test, y_test) #negative because we want to maximize the score
    
if __name__ == '__main__':

    
    # I. make training data
    print("Making training data")
    training_data = make_training_data('leafy spurge', SNR_min, SNR_max)
    # consolidate training data

    print("Consolidating training data")

    all_X_val = {}
    training, training_class, validation_veg_comp, category_name = training_data
    
    #scale the training data!
    scaler = preprocessing.StandardScaler().fit(training)
    training = scaler.transform(training)
    
    #validation dataset
    X_val_df = pd.DataFrame(training,columns=['B','V','R','I'])
    #true value
    X_val_df['validation'] = training_class    
    #water %
    X_val_df['water'] = validation_veg_comp
    all_X_val['leafy spurge'] = X_val_df    
        
    all_categories = list(special_read_list.keys()) + ['all']

    # II. calculate alpha for binary classification

    print("Finding best alpha for binary classification")
    alpha_dict['leafy spurge'] = get_alpha('leafy spurge', training,
                                array(all_X_val['leafy spurge']['validation']))
    print('Best alpha\n',alpha_dict)

    #save the data
    pk.dump(alpha_dict, 
        open('{}/pruning_alpha.pk'.format(binary_output_dir), 'wb'))

    #III. calculate alpha for multiclass classification
    print("Finding best alpha for multiclass classification")
    alpha_dict['leafy spurge'] = get_alpha('leafy spurge', training,
                                array(all_X_val['leafy spurge']['water']))
    print('Best alpha\n',alpha_dict)

    #save the data
    pk.dump(alpha_dict, 
        open('{}/pruning_alpha.pk'.format(multiclass_output_dir), 'wb'))