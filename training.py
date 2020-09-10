"""
    Author: Dang Pham (contact: dcp237@cornell.edu)
    Last update: March 11, 2020

    Terminal arguments:
        Required:
            -mpc: number of cores. Recommended: 1, 7, 14, 21, etc.
            -t: training type. Binary of multiclass
        Optional:
            -o: output directory. Default is "default".
            -smin: minimum SNR to train on. Default is 1.
            -smax: maxmimum SNR. Default is 100.
            -s: seed for random state. Default is 10.
            -k: number of k-folds. Default is 10.
            -ks: Shuffle dataset during k-fold? Default is True.
            -a: pruning alpha file for CART trees. Default is "default".
                The pruning alpha file is the output of 
                find_decision_tree_depth.py

    Description: This code performs the following:
        1) Creates noise photometric data from available simulated spectra.

           Note that the Signal to noise ratio is taken randomly from a uniform
           distribution in the range [SNR_min, SNR_max].

           The data are automatically scaled using sklearn.preprocessing. The
           scaler is saved and can be used to scale new data!

        2) Train LR, LDA, KNN, CART, NB, SVM, MVH, and MVS models on the
           data. The ML models are trained to perform binary and multiclass
           classification on biota. Which are True/False detection of biota
           existence or classifying which 5% bin of vegetation the data belongs
           to.
           
           Furthermore, the trained models can be used to
           predict any data with a similar noise model and if the SNR of the
           data is within the range [SNR_min, SNR_max].

        3) k-Fold (default k=10) cross validation is used.
        
        4) The code is written to be able to use many cores. Set the parameter
           cores = 1 for a fully serialized code. Parallelization makes the
           program runs (much) faster! Recommended cores: anything in multiples
           of 7 cores (e.g. 7, 14, 35).

    Version requirement: sklearn version >= 0.22.1. Python 3.
"""


#core imports for numerical, plotting, data management, and IO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import pandas as pd

#for multiprocessing
from joblib import Parallel, delayed

#for k-fold
from sklearn import model_selection
#for the accuracy score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
#for preprocessing the data
from sklearn import preprocessing
#the ML models we are using (single methods)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#ensemble method
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

#important helper methods
import main_methods
#import standard BVRI filters
import bvri_filters

#miscellaneous important things
from copy import copy, deepcopy
import time
from datetime import datetime
import glob
import argparse
import logging

plt.switch_backend('Agg')
DEFAULT_INPUT_ML_DATA_PATH = 'data/ML_data_cloud_rayleigh/ML_'
#all biota and their spectra path
special_read_list = {
                     'leafy spurge': 'spectra/USGS_Vegetation_and_Microorganisms/LeafySpurge/leafyspurge_spurge-a2-jun98.11306.asc'
                    }

################################# ARGUMENTS ##################################

#process arguments from the terminal command
try:
    parser = argparse.ArgumentParser()
except:
    argparse.print_help()
    sys.exit(0)

#required arguments
parser.add_argument('-t', '--ttype', type=str, required=True,
                    choices = ['binary', 'multiclass'],
                    help='Training types: Multiclass or binary training')
parser.add_argument('-mpc', '--cores',  type=int, default=9, required=True,
                    help='How many cores for multiprocessing? (multiples of 9 are best)' )
#optional arguments
parser.add_argument('-io', '--io', type=str,
                    default = 'default',
                    help='In/output directory' )
parser.add_argument('-l', '--log', type=str, default='default',
                    help='Logging directory')
parser.add_argument('-smin', '--snrmin', type=int, default=1,
                    help='Minimum SNR to train' )
parser.add_argument('-smax', '---snrmax',  type=int, default=100,
                    help='Maximum SNR to train' )
parser.add_argument('-s', '--seed', type=int, default=10,
                help='Seed for the random state. Set to None for full random' )
parser.add_argument('-k', '--kfold', type=int, default=10,
                    help='K in k-fold' )
parser.add_argument('-ks', '--kshuffle', type=bool, default=True,
                    help='Shuffle while doing K-fold?' )
args = parser.parse_args()

training_mode = args.ttype
SNR_max = args.snrmax                     
SNR_min = args.snrmin                       
cores = args.cores                       
seed = args.seed                       
kfold_K = args.kfold                     
kfold_shuffle = args.kshuffle
######################### SETTING ARGUMENTS ####################################

#setting the io directory
if args.io == 'default':
    if training_mode == 'binary':
        io_dir = 'output/binary_class/pickles_data'
    else:
        io_dir = 'output/multiclass/pickles_data'
else:
    io_dir = args.io
#set the hyperparameter files
hyper_file = f'{io_dir}/hyperparams.pk'
#set the seed
np.random.seed(seed) 
#get the hyperparameters
hyperparams = pk.load(open(hyper_file, 'rb'))

#recommended number of cores
factors = lambda n: [ i for i in range(1, n + 1) if n % i == 0]
print(f'Recommended cores: {factors(9*kfold_K)}')
print(f'You are running with {cores} cores')
cores = int(input('Press ENTER to confirm or input the new number of cores : ') 
         or cores)


#timing
time_of_run = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
initial_time = time.time()
################################## LOGGING ####################################

#create logging filename
if args.log == 'default':
    if training_mode == 'binary':
        log_dir = 'output/binary_class/logs'
    else:
        log_dir = 'output/multiclass/logs'
else:
    log_dir = args.log
    
current_logcounter = len(glob.glob(f'{log_dir}/training_*.log'))+1
log_filename = f'{log_dir}/training_{current_logcounter}.log'


#set up logging to file 
logger = logging.getLogger('logs')
logger.setLevel(logging.DEBUG)
logger.propagate = False
handler = logging.FileHandler(log_filename)
formatter = logging.Formatter(
        '[%(asctime)s|PID %(process)s|%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#log some basic info
logger.info(f'Start time: {time_of_run}')
logger.info(f'Running {training_mode} classification on noisy data')
logger.info(f'Cores: {cores}')
logger.info(f'S/N range: [{SNR_min},{SNR_max}]')
logger.info(f'Seed: {seed}')
logger.info(f'Hyperparameters: {hyper_file}')
logger.info(f'K-Fold: {kfold_K}-fold')
logger.info(f'K-Fold shuffle: {kfold_shuffle}')
logger.info(f'Output folder: {io_dir}\n')

#work around to get logger inside parallel
def get_logger():
    # set up the logger
    logger = logging.getLogger('logs')
    # prevent hierarchy propagation (duplicate)
    logger.propagate = False
    if not logger.handlers:
        # configure the logging properties
        handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter(
                '[%(asctime)s|PID %(process)s|%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        # set the logging properties
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    
    return logger

print(f'\nStart time: {time_of_run}')
print(f'Output directory: {io_dir}')
print(f'Check {log_filename} for run log')

######################## HELPER FUNCTIONS FOR K-FOLD ##########################
# make_training_data_snr_range: make training data given snr range
# train_save: train models and save

def make_training_data_snr_range(category_name, SNR_min, SNR_max, mode):
    
    logger = get_logger()
    category_df = pd.read_pickle(DEFAULT_INPUT_ML_DATA_PATH+
                        category_name.replace(' ','')+'.pkl')

    logger.info(f'Creating noisy data for {category_name}')
    
    #make color from spectra
    df_training_i = main_methods.all_spectra_to_flux( 
        {'training':category_df}, bvri_filters.i_filter_profile, 
        bvri_filters.i_interpolated, error_report=False)

    df_training_b = main_methods.all_spectra_to_flux( 
        {'training':category_df}, bvri_filters.b_filter_profile, 
        bvri_filters.b_interpolated, error_report=False)

    df_training_v = main_methods.all_spectra_to_flux( 
        {'training':category_df}, bvri_filters.v_filter_profile, 
        bvri_filters.v_interpolated, error_report=False)

    df_training_r = main_methods.all_spectra_to_flux( 
        {'training':category_df}, bvri_filters.r_filter_profile, 
        bvri_filters.r_interpolated, error_report=False)

    df_training_b_0 = np.array(df_training_b[0][1:])
    df_training_v_0 = np.array(df_training_v[0][1:])
    df_training_r_0 = np.array(df_training_r[0][1:])
    df_training_i_0 = np.array(df_training_i[0][1:])
    
    snr = np.random.uniform(low=SNR_min, high=SNR_max, 
                            size=len(df_training_b_0))

    df_training_b = df_training_b_0 + main_methods.add_noise(snr, 
        df_training_b_0)
    df_training_v = df_training_v_0 + main_methods.add_noise(snr, 
        df_training_v_0)
    df_training_r = df_training_r_0 + main_methods.add_noise(snr, 
        df_training_r_0)
    df_training_i = df_training_i_0 + main_methods.add_noise(snr, 
        df_training_i_0)
    
    category_df['B'] = df_training_b
    category_df['V'] = df_training_v
    category_df['R'] = df_training_r
    category_df['I'] = df_training_i

    #make classifier for supervised training
    if mode == 'binary':
        training_class = []
        for i in category_df['classification']:
            if i['seawater'] > 0:
                training_class.append(1) #true, vegetation
            elif i['seawater'] == 0:
                training_class.append(0) #false, no vegetation
            else:
                raise ValueError
    elif mode == 'multiclass':
        training_class = [i['seawater'] for i in category_df['classification']]

    training = []
    for i in range(len(df_training_i)):
        training.append([df_training_b[i], df_training_v[i],
                         df_training_r[i], df_training_i[i]])
    #vegetation %
    validation_veg_comp = []

    for i in range(len(category_df['B'])):
        veg = category_df.iloc[i]['classification']['seawater']
        validation_veg_comp.append(veg)

    training_class = np.array(training_class)
    training = np.array(training)
    validation_veg_comp = np.array(validation_veg_comp)
    
    return training, training_class, validation_veg_comp, category_name

def __train_save_helper(name, model, X_train, Y_train, kfold_count,
                        X_val_df, X_train_df):
    logger = get_logger()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_validation)
    ba_score = balanced_accuracy_score(Y_validation, predictions)

    X_val_df[name] = predictions
    X_train_df[name] = model.predict(X_train)

    logger.info(f'Finished training (k={kfold_count}, {name})')

    return name, ba_score

def train_save(X_train, X_validation, Y_train,Y_validation, 
               veg_train, veg_validation, models, kfold_count):

    logger = get_logger()
    logger.info(f'(k={kfold_count}) Begin training models (t={time.time() - initial_time})')
    
    #train the models

    #make validation dataset
    X_val_df = pd.DataFrame(X_validation,columns=['B','V','R','I'])
    X_val_df['validation'] = Y_validation
    X_val_df['vegetation'] = veg_validation

    #make training dataset
    X_train_df = pd.DataFrame(X_train,columns=['B','V','R','I'])
    X_train_df['validation'] = Y_train
    X_train_df['vegetation'] = veg_train


    #traing the models using parallel processing
    Parallel(n_jobs = cores, pre_dispatch=9)(delayed(__train_save_helper)(
        name, model, X_train, Y_train, kfold_count, X_val_df, X_train_df)
        for name, model in models    
    )
    logger.info(f'(k={kfold_count}) Training completed (t={time.time() - initial_time})')

    #save the models
    filepath = f'{io_dir}/models_{kfold_count}.pk'
    pk.dump(models, open(filepath,'wb'))
    logger.info(f'(k={kfold_count}): Models saved to {io_dir}')
    
    #save the training/validation dataset
    filepath = f'{io_dir}/validation_data_{kfold_count}.pk'
    pk.dump(X_val_df, open(filepath,'wb'))
    logger.info(f'(k={kfold_count}): Validation data saved to {io_dir}')
    
    filepath = f'{io_dir}/training_data_{kfold_count}.pk'
    pk.dump(X_train_df, open(filepath,'wb'))
    logger.info(f'(k={kfold_count}): Training data saved to {io_dir}')

############################## MAIN CODE RUN ##################################

if __name__ == '__main__':
    logger.info(f'Filters and Atmosphere Transmission imported (t={time.time() - initial_time})')

    ### MAKE TRAINING DATA ###
    logger.info(f'Creating training data (t={time.time() - initial_time})')

    #make training data using parallel processing
    tmp_training_data = Parallel(n_jobs=cores, pre_dispatch=7)(delayed(
        make_training_data_snr_range)(
        category_name, SNR_min, SNR_max, training_mode) 
        for category_name in special_read_list.keys())

    #training data comprises of all 6 types of biota
    all_training       = [] #training data point, BVRI colors
    all_training_class = [] #classification (e.g. 0, 1 for binary class.)
    all_validation_veg = [] #percentage of vegetation

    #combine the parallel processing data
    for training_data in tmp_training_data:
        training = training_data[0]
        training_class = training_data[1]
        validation_veg_comp = training_data[2]
        
        all_training += list(training)
        all_training_class += list(training_class)
        all_validation_veg += list(validation_veg_comp)
        
    #scale the training data
    scaler = preprocessing.StandardScaler().fit(all_training)
    all_training = scaler.transform(all_training)

    #make them np np.array
    all_training = np.array(all_training)
    all_training_class = np.array(all_training_class)
    all_validation_veg = np.array(all_validation_veg)

    #save the scaler for using on future data
    pk.dump(scaler, open(f'{io_dir}/scaler.pk', 'wb'))
    logger.info(f'Scaler saved {io_dir}/scaler.pk')
    logger.info(f'Finished creating data (t={time.time() - initial_time})\n')

    ### MAKE TRAINING MODELS ###

    #make the training models template models_0
    models_0 = []

    if training_mode == 'binary':
        models_0.append(('LR', LogisticRegression(solver='saga',
            random_state=seed)))
    elif training_mode =='multiclass':
        #multiclass needs more LR iterations to find convergent solution
        models_0.append(('LR', LogisticRegression(solver='saga',
        random_state=seed, multi_class='multinomial', 
        max_iter = 10000, n_jobs = cores)))
    
    logger.info(f'Hyperparameters info')
    knn_param = hyperparams['knn_param']['n_neighbors']
    logger.info(f'KNN n_neighbors: {knn_param}')
    svm_kernel = hyperparams['svm_param']['kernel']
    logger.info(f'SVM Kernel: {svm_kernel}')
    RF_param = hyperparams['rf_param']['n_estimators']
    logger.info(f'RF Trees: {RF_param}')
    alpha = hyperparams['cartparam']['leafy spurge']
    logger.info(f'CART Alpha: {alpha}\n')
    
    models_0.append(('LDA', LinearDiscriminantAnalysis()))
    models_0.append(('KNN', KNeighborsClassifier(n_neighbors=knn_param)))
    models_0.append(('CART', DecisionTreeClassifier(random_state=seed, 
        ccp_alpha = alpha)))
    models_0.append(('NB', GaussianNB()))
    models_0.append(('SVM', SVC(kernel = svm_kernel, 
                                probability=True,gamma='scale',
                                random_state=seed)))
    models_0.append(('RF', RandomForestClassifier(n_estimators=RF_param,
        random_state=seed, ccp_alpha = alpha)))
    #put all models together
    single_models=copy(models_0)
    #now add on the voting classifiers
    models_0.append(('MVH', VotingClassifier(estimators=single_models,
        voting='hard')))
    models_0.append(('MVS', VotingClassifier(estimators=single_models,
        voting='soft')))
    
    logger.info(f'{kfold_K}-fold begins (t={time.time() - initial_time})')

    ### PERFORM K-FOLD SPLIT AND TRAINING ON THE MODELS ###
    kfold = model_selection.KFold(n_splits = kfold_K,
                                shuffle=kfold_shuffle,
                                random_state=seed)

    #set up for parallel processing
    args_for_mp = []
    kfold_count = 1
    for training_index, validation_index in kfold.split(all_training):
        X_train        = all_training[training_index]
        X_validation   = all_training[validation_index]

        Y_train        = all_training_class[training_index] 
        Y_validation   = all_training_class[validation_index]

        veg_train      = all_validation_veg[training_index]
        veg_validation = all_validation_veg[validation_index]
        
        logger.info(f'Training and validation dataset created (t={time.time() - initial_time})')
        logger.info(f'Training dataset length: {len(X_train)}')
        logger.info(f'Validation dataset length: {len(X_validation)}')

        models = deepcopy(models_0)
        args_for_mp.append((X_train, X_validation, Y_train, Y_validation, 
                            veg_train, veg_validation, models, kfold_count))
        kfold_count += 1

    #parallel processing
    Parallel(n_jobs=cores, verbose=10)(delayed(train_save)(
            X_train, X_validation, Y_train,Y_validation, 
               veg_train, veg_validation, models, kfold_count) 
            for X_train, X_validation, Y_train,Y_validation, 
               veg_train, veg_validation, models, kfold_count 
            in args_for_mp)

    ### SAVE DATA ###
    for i in np.arange(1,kfold_K+1):
        filepath = f'{io_dir}/validation_data_{i}.pk'
        X_val_df = pk.load(open(filepath, 'rb'))
        logger.info(f'k={i} model saved to {filepath}')

    ## DONE! ###
    logger.info(f'Total runtime: {time.time() - initial_time} seconds')
    print(f'Total runtime: {time.time() - initial_time} seconds')
    
    end_time = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    logger.info(f'End time: {end_time}')
    print(f'End time: {end_time}')