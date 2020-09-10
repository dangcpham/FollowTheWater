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

#hyperparameter tuning
from sklearn.model_selection import GridSearchCV

#the ML models we are using (single methods)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
#ensemble method
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

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
import glob
import logging
from copy import copy, deepcopy
from decimal import Decimal
import argparse

special_read_list = {
                     'leafy spurge': 'spectra/USGS_Vegetation_and_Microorganisms/LeafySpurge/leafyspurge_spurge-a2-jun98.11306.asc'
                    }
MAX_ITER = 100
DEFAULT_INPUT_ML_DATA_PATH = 'data/ML_data_cloud_rayleigh/ML_'

################################# ARGUMENTS ##################################
#process arguments from the terminal command
try:
    parser = argparse.ArgumentParser()
except:
    argparse.print_help()
    sys.exit(0)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--ttype', type=str, required=True,
                    choices = ['binary', 'multiclass'],
                    help='Training types: Multiclass or binary training')
parser.add_argument('-mpc', '--cores',  type=int, default=6, required=True,
                    help='How many cores for multiprocessing?' )

parser.add_argument('-o', '--output', type=str, default = 'default',
                    help='Output directory')
parser.add_argument('-l', '--log', type=str, default='default',
                    help='Logging directory')
parser.add_argument('-smin', '--snrmin', type=int, default=1,
                    help='Minimum SNR to train' )
parser.add_argument('-smax', '---snrmax',  type=int, default=100,
                    help='Maximum SNR to train' )
parser.add_argument('-s', '--seed', type=int, default=10,
                help='Seed for the random state. Set to None for full random' )

args = parser.parse_args()

training_mode = args.ttype
SNR_max = args.snrmax                     
SNR_min = args.snrmin                       
cores = args.cores                       
seed = args.seed 

#setting the output directory
if args.output == 'default':
    if training_mode == 'binary':
        output_dir = 'output/binary_class/pickles_data'
    else:
        output_dir = 'output/multiclass/pickles_data'
else:
    output_dir = args.output

                    
time_of_run = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
initial_time = time.time()
np.random.seed(seed)


################################## LOGGING ####################################

#create logging filename
if args.log == 'default':
    if training_mode == 'binary':
        log_dir = 'output/binary_class/logs'
    else:
        log_dir = 'output/multiclass/logs'
else:
    log_dir = args.log
    
current_logcounter = len(glob.glob(f'{log_dir}/hyperparams_*.log'))+1
log_filename = f'{log_dir}/hyperparams_{current_logcounter}.log'


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
logger.info(f'Finding hyperparameters for {training_mode} classification')
logger.info(f'Using {cores} cores')
logger.info(f'Seed: {seed}')
logger.info(f'S/N range: [{SNR_min}, {SNR_max}]')
logger.info(f'Output folder: {output_dir}')

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

print(f'Start time: {time_of_run}')
print(f'Output directory: {output_dir}')
print(f'Check {log_filename} for run log')

###############################################################################

### HELPER FUNCTIONS FOR MULTIPROCESSING ###
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

def DecisionTree_find_alpha_helper(ccp_alpha,X_train, y_train):
    """
        Helper function for get_alpha()
    """

    return DecisionTreeClassifier(random_state=seed, ccp_alpha=ccp_alpha).fit(
            X_train, y_train)

def get_alpha(category_name, training, training_class):
    """
        Code is from 
        https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
    """
    X_train, X_test, y_train, y_test = train_test_split(training, 
                                            training_class, random_state=seed)
    
    clf = DecisionTreeClassifier(random_state=seed)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    clfs = []

    select_list = list(arange(0, len(ccp_alphas), 100))
    select_list = unique(array(select_list))
    ccp_alphas, impurities = ccp_alphas[select_list], impurities[select_list]

    clfs = Parallel(n_jobs=cores)(delayed(
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

    return res.x

def opt_func(x, X_train, y_train, X_test, y_test):
    """
        Objective function to minimize for get_alphas
    """
    
    #ccp_alpha must be > 0
    if x < 0:
        return np.inf
    
    clf = DecisionTreeClassifier(random_state=10, ccp_alpha=x)
    clf.fit(X_train, y_train)
    return -clf.score(X_test, y_test) #negative because we want to maximize the score

def print_clf_results(clf):
    """
        Print gridsearch results.
    """
    logger = get_logger()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        logger.info('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))

def rf_search(x0, dx, training, training_class):
    """
        Find the best number of trees in the range [x0 - dx, x0 + dx].
        If x0 - dx < 0, then the range becomes [0, x0 + dx]
    """

    #get the list of trees
    no_of_trees = np.arange(x0 - dx, x0 + dx + 1, 1)
    #make sure it is >0
    no_of_trees = no_of_trees[no_of_trees > 0]
    #set the parameters to grid search
    parameters = {'n_estimators': no_of_trees}
    #set the classifier
    rf = RandomForestClassifier(random_state=seed, ccp_alpha = alpha)
    #perform the gridsearch
    clf = GridSearchCV(rf, parameters, n_jobs=cores, scoring='balanced_accuracy', cv=3)
    clf.fit(training, training_class)
    
    print_clf_results(clf)
    return clf.best_estimator_.n_estimators

def look_around(f, x0, dx, training, training_class): 
    """
        Finds locally optimal, exact solution to integer hyperparameters.
    """
    logger = get_logger()
    # function f looks for best integer solution in range [x0 - dx, x0 + dx].
    # function rf_search does this
    x1 = f(x0, dx, training, training_class)
    logger.info(f'Searching in range {x0} +\- {dx}')
    # if the value is not at the end points, then this is locally maximal
    if x0 - dx < x1 < x0 + dx:
        return x1
    # if not, then look at the end point and perform this again
    elif (x1 == x0 - dx) or (x1 == x0 + dx):
        return look_around(f, x1, dx, training, training_class)


if __name__ == '__main__':

    
    # I. make training data
    logger.info('Making training data')
    args_for_mp = [(category_name, SNR_min, SNR_max) 
                    for category_name in special_read_list.keys()]

    tmp_training_data = Parallel(n_jobs=cores)(delayed(
        make_training_data_snr_range)(
        category_name, SNR_min, SNR_max, training_mode) 
        for category_name, SNR_min, SNR_max in args_for_mp)

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

    all_categories = list(special_read_list.keys())

    # II. Find the best alpha for CART trees

    logger.info('Finding best alpha for CART')

    args_for_mp = [(category_name, all_training,all_training_class)
                    for category_name in all_categories]
    
    tmp_alpha_list = Parallel(n_jobs = cores)(delayed(
        get_alpha)(category_name, training, training_class) for 
        category_name, training, training_class in args_for_mp
    )

    alpha_dict = {}
    for i, category_name in enumerate(all_categories):
        alpha_dict[category_name] = tmp_alpha_list[i]

    logger.info(f'Best CART alpha: {alpha_dict["leafy spurge"]}')
    alpha = alpha_dict['leafy spurge']

    #save the data
    #pk.dump(alpha_dict, open('{}/pruning_alpha.pk'.format(binary_output_dir), 'wb'))

    # III. find the best number of RF trees
    #number of estimators to test, log spacing
    no_of_trees = np.geomspace(1, 400, 50)

    #round to integers
    no_of_trees = list(map(int, np.floor(no_of_trees)))
    no_of_trees = np.unique(np.array(no_of_trees))
    logger.info(f'RF Trees to search: {no_of_trees}. Length: {len(no_of_trees)}')

    #make the RF models
    RF_models = [(n, RandomForestClassifier(n_estimators=n,
            random_state=seed, ccp_alpha = alpha)) for n in no_of_trees]
    
    #gridsearch the trees
    #rf and svm takes a long time, so sample a section
    X_train, X_test, y_train, y_test = train_test_split(all_training, 
                        all_training_class, test_size=0.1, random_state=seed)
    parameters = {'n_estimators': no_of_trees}
    rf = RandomForestClassifier(random_state=seed, ccp_alpha = alpha)
    rf_clf = GridSearchCV(rf, parameters, n_jobs=cores, scoring='balanced_accuracy', cv=3)
    rf_clf.fit(X_test, y_test)

    print_clf_results(rf_clf)

    #refine the search
    best_tree = {'n_estimators': look_around(rf_search, rf_clf.best_estimator_.n_estimators, 10, X_test, y_test)}
    logger.info(f'RF Optimal Trees: {best_tree["n_estimators"]}')

    # IV. Find best SVM Kernel
    kernels_search = ('linear', 'poly', 'rbf', 'sigmoid')
    logger.info(f'SVM Kernels to search: {kernels_search}')

    parameters = {'kernel': kernels_search}
    svm = SVC(gamma='scale', random_state=seed, cache_size = 1000)
    svm_clf = GridSearchCV(svm, parameters, n_jobs=cores, scoring='balanced_accuracy', cv=3)
    
    svm_clf.fit(X_test, y_test)
    best_svm_param = svm_clf.best_params_

    print_clf_results(svm_clf)
    logger.info(f'SVM Optimal Kernel: {best_svm_param["kernel"]}')

    # V. Find best number of neighbors for KNN
    #number of neighbors in KNN, linear spacing
    no_of_neighbors = np.arange(1, 50, 1)
    logger.info(f'KNN Neighbors to search: {no_of_neighbors}')

    parameters = {'n_neighbors': no_of_neighbors}
    knn =  KNeighborsClassifier()
    knn_clf = GridSearchCV(knn, parameters, n_jobs=10, scoring='balanced_accuracy', cv=3)
    knn_clf.fit(all_training, all_training_class)
    best_knn_param = knn_clf.best_params_

    print_clf_results(knn_clf)
    logger.info(f'KNN Optimal Neighbors: {best_knn_param["n_neighbors"]}')

    # VI. Save output
    output = {'rf_param' : best_tree,
             'rf_clf'   : rf_clf,
             'svm_param': best_svm_param,
             'svm_clf'  : svm_clf,
             'knn_param': best_knn_param,
             'knn_clf'  : knn_clf,
             'cart_param': alpha_dict}

    pk.dump(output, open(f'{output_dir}/hyperparams.pk', 'wb'))

    ## DONE! ###
    logger.info(f'Total runtime: {time.time() - initial_time} seconds')
    print(f'Total runtime: {time.time() - initial_time} seconds')
    
    end_time = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    logger.info(f'End time: {end_time}')
    print(f'End time: {end_time}')