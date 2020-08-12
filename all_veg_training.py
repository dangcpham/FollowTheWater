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
from sklearn.ensemble import VotingClassifier

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
special_read_list = {'agrococcus':'spectra/Hegde_2013/Agrococcus/Agrococcus%20sp._KM349956.csv',
                     'geodermatophilus': 'spectra/Hegde_2013/Geodermatophilus/Geodermatophilus%20sp._KM349882.csv',
                     'bark': 'spectra/ECOSTRESS_nonphotosyntheticvegetation/bark/nonphotosyntheticvegetation.bark.acer.rubrum.vswir.acru-1-81.ucsb.asd.spectrum.csv',
                     'lichen': 'spectra/ECOSTRESS_nonphotosyntheticvegetation/lichen/nonphotosyntheticvegetation.lichen.lichen.species.vswir.vh298.ucsb.asd.spectrum.csv',
                     'aspen': 'spectra/USGS_Vegetation_and_Microorganisms/Aspen_Leaf-A/aspen_leaf_dw92-2.10753.asc',
                     'leafy spurge': 'spectra/USGS_Vegetation_and_Microorganisms/LeafySpurge/leafyspurge_spurge-a2-jun98.11306.asc'
                    }

################################# PARAMETERS ##################################

#process arguments from the terminal command
parser = argparse.ArgumentParser()
#required arguments
parser.add_argument('-t', '--ttype', type=str, required=True,
                    choices = ['binary', 'multiclass'],
                    help='Training types: Multiclass or binary training')
parser.add_argument('-mpc', '--cores',  type=int, default=6, required=True,
                    help='How many cores for multiprocessing?' )
#optinal arguments
parser.add_argument('-o', '--output', type=str,
                    default = 'default',
                    help='Output directory' )
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
parser.add_argument('-a', '--alpha', type=str, 
                    default='default',
                    help='List of alpha used for pruning CART tree')
args = parser.parse_args()

training_mode = args.ttype
SNR_max = args.snrmax                     
SNR_min = args.snrmin                       
cores = args.cores                       
seed = args.seed                       
kfold_K = args.kfold                     
kfold_shuffle = args.kshuffle

if args.output == 'default':
    if training_mode == 'binary':
        output_dir = 'output/binary_class/all'
    else:
        output_dir = 'output/multiclass/all'
else:
    output_dir = args.output

if args.alpha == 'default':
    if training_mode == 'binary':
        alpha_file = 'output/binary_class/pruning_alpha.pk'
    else:
        alpha_file = 'output/multiclass/pruning_alpha.pk'
else:
    alpha_file = args.alpha

#set the seed
np.random.seed(seed) 
#get the pruning alpha for this particular dataset
alpha_dict = pk.load(open(alpha_file, 'rb'))
alpha = alpha_dict['all']
#timing
time_of_run = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
initial_time = time.time()

print('Training on all vegetations')
print('Output directory: {}'.format(output_dir))

################################## LOGGING ####################################
#set up logging to file 

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename=output_dir + '/run_{}.log'.format(
                        len(glob.glob(output_dir + '/*.log'))+1),
                    filemode='w')

logging.info("Running binary classification for noisy data")
logging.info("All vegetations")
logging.info("Cores: {}".format(cores))
logging.info("SNR range: [{},{}]".format(SNR_min,SNR_max))
logging.info("Seed: {}".format(seed))
logging.info("Pruning alpha: {}".format(alpha))
logging.info("K-Fold: {}-fold".format(kfold_K))
logging.info("K-Fold shuffle: {}".format(kfold_shuffle))
logging.info("Output folder: {}".format(output_dir))
logging.info("Run start at {}\n".format(time_of_run))

######################## HELPER FUNCTIONS FOR K-FOLD ##########################
# make_training_data_snr_range: make training data given snr range
# train_save: train models and save
# save_extra_plot: save plots

def make_training_data_snr_range(category_name, SNR_min, SNR_max, mode):
    category_df = pd.read_pickle(DEFAULT_INPUT_ML_DATA_PATH+
                        category_name.replace(' ','')+'.pkl')

    print('Creating noisy data for {} (t={})'.format(category_name, 
                                            time.time() - initial_time))

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
            if i['tree'] > 0:
                training_class.append(1) #true, vegetation
            elif i['tree'] == 0:
                training_class.append(0) #false, no vegetation
            else:
                raise ValueError
    elif mode == 'multiclass':
        training_class = [i['tree'] for i in category_df['classification']]

    training = []
    for i in range(len(df_training_i)):
        training.append([df_training_b[i], df_training_v[i],
                         df_training_r[i], df_training_i[i]])
    #vegetation %
    validation_veg_comp = []

    for i in range(len(category_df['B'])):
        veg = category_df.iloc[i]['classification']['tree']
        validation_veg_comp.append(veg)

    training_class = np.array(training_class)
    training = np.array(training)
    validation_veg_comp = np.array(validation_veg_comp)
    
    return training, training_class, validation_veg_comp, category_name

def __train_save_helper(name, model, X_train, Y_train, kfold_count,
                        X_val_df, X_train_df):
    print(kfold_count, name)
    model.fit(X_train, Y_train)
    #print('(k={}) {} trained'.format(kfold_count, name))

    predictions = model.predict(X_validation)
    ba_score = balanced_accuracy_score(Y_validation, predictions)

    X_val_df[name] = predictions
    X_train_df[name] = model.predict(X_train)

    print('(k={}) {}: {}'.format(kfold_count, name, ba_score))
    
    return name, balanced_accuracy_score(Y_validation, predictions)

def train_save(X_train, X_validation, Y_train,Y_validation, 
               veg_train, veg_validation, models, kfold_count):
    
    print("Fold run number: {}".format(kfold_count))   

    #train the model
    print("(k={}) Begin training models (t={})".format(
        kfold_count, time.time() - initial_time))

    #make validation dataset
    X_val_df = pd.DataFrame(X_validation,columns=["B","V","R","I"])
    X_val_df['validation'] = Y_validation
    X_val_df['vegetation'] = veg_validation

    #make training dataset
    X_train_df = pd.DataFrame(X_train,columns=["B","V","R","I"])
    X_train_df['validation'] = Y_train
    X_train_df['vegetation'] = veg_train

    #traing the models using parallel processing
    Parallel(n_jobs = cores, verbose = 10)(delayed(__train_save_helper)(
        name, model, X_train, Y_train, kfold_count, X_val_df, X_train_df)
        for name, model in models    
    )
    print("(k={}) Training completed (t={})".format(
        kfold_count, time.time() - initial_time))

    #save the models
    filepath = output_dir + '/models_{}.pk'.format(kfold_count)
    pk.dump(models, open(filepath,'wb'))
    print("KFold run {} completed (t={})".format(
        kfold_count, time.time() - initial_time))    
    
    #save the training/validation dataset
    print("(k={}) Saving Validation Data".format(kfold_count))
    filepath = output_dir + '/validation_data_{}.pk'.format(kfold_count)
    pk.dump(X_val_df, open(filepath,'wb'))

    print("(k={}) Saving Training Data".format(kfold_count))
    filepath = output_dir + '/training_data_{}.pk'.format(kfold_count)
    pk.dump(X_train_df, open(filepath,'wb'))

def save_extra_plot(X_val_df, kfold_count):
    ### PLOT THE RESULTS ###
    fig, axes = plt.subplots(nrows=3,ncols=3,
                        sharex=True,sharey=True,figsize=(10,10))
    axes = axes.flat

    bins_list = np.arange(0,105,5)
    xticks_list = np.arange(0,105,10)

    i=0
    for name, _ in models:
        ax = axes[i]

        ax.hist(X_val_df.loc[X_val_df[name] == X_val_df['validation']]['vegetation'],
            bins=bins_list, label='Accurate', histtype='bar', 
            stacked=True, fill=False)

        ax.hist(X_val_df.loc[X_val_df[name] != X_val_df['validation']]['vegetation'],
            bins=bins_list,label='Inaccurate',histtype='step', 
            stacked=True, fill=True,alpha=0.7)
        
        ax.set_xticks(xticks_list)
        
        ax.set_title('{}\nBA:{}, A:{}'.format(
            name,
            round(balanced_accuracy_score(X_val_df['validation'],
                X_val_df[name]),2),
            round(accuracy_score(X_val_df['validation'],
                X_val_df[name]),2)
        ),y=0.8)
        i+=1

    axes[5].legend(loc='center right')
    axes[-1].set_xlabel('Vegetation %')
    axes[-2].set_xlabel('Vegetation %')
    fig.suptitle('Binary classification results',y=0.92)
    plt.subplots_adjust(wspace=0,hspace=0)
    fig.savefig(output_dir + '/class_res_hist_{}.pdf'.format(kfold_count))
    plt.close(fig)

############################## MAIN CODE RUN ##################################

if __name__ == '__main__':
    logging.info('Filters and Atmosphere Transmission imported (t={})'.format(
        time.time() - initial_time))

    ### MAKE TRAINING DATA ###

    logging.info('Making training data')

    #make training data using parallel processing
    tmp_training_data = Parallel(n_jobs=cores,verbose=10)(delayed(
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
    pk.dump(scaler, open(output_dir + '/scaler.pk', 'wb'))
    logging.info('Scaler saved (t={})'.format(time.time() - initial_time))

    logging.info('Training data created')

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
    
    models_0.append(('LDA', LinearDiscriminantAnalysis()))
    models_0.append(('KNN', KNeighborsClassifier()))
    models_0.append(('CART', DecisionTreeClassifier(random_state=seed, 
        ccp_alpha = alpha)))
    models_0.append(('NB', GaussianNB()))
    models_0.append(('SVM', SVC(probability=True,gamma='scale',
        random_state=seed)))
    #put all single models together
    single_models=copy(models_0)

    #now add on the ensemble methods
    models_0.append(('MVH', VotingClassifier(estimators=single_models,
        voting='hard')))
    models_0.append(('MVS', VotingClassifier(estimators=single_models,
        voting='soft')))

    logging.info("Initial training models created (t={})".format(
        time.time() - initial_time))
    
    logging.info("{}-fold begins (t={})".format(kfold_K, 
        time.time() - initial_time))

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
        
        logging.info("Training and validation " +
            "dataset created (t={})".format(time.time() - initial_time))
        logging.info("Training dataset length: {}".format(len(X_train)))
        logging.info("Validation dataset length: {}".format(len(X_validation)))

        #prepare for paralleling
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
        X_val_df = pk.load(
            open(output_dir + '/validation_data_{}.pk'.format(i), 'rb')
        )
        save_extra_plot(X_val_df, i)

    ## DONE! ###
    logging.info("Total running time: {}".format(time.time() - initial_time))
    end_time = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    logging.info("End time: {}".format(end_time))