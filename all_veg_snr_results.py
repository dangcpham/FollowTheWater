#core imports for numerical, plotting, data management, and IO
from numpy import *
from matplotlib.pyplot import *
from matplotlib import rcParams
import pandas as pd
import pickle as pk
from matplotlib import cm
import seaborn as sns

#for the accuracy score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_curve, auc, confusion_matrix

#for preprocessing the data
from sklearn import preprocessing

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

#important helper methods
import main_methods
#import standard BVRI filters
import bvri_filters

#for parallel processing
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

switch_backend('Agg')
#rcParams["font.size"] = 16
#rcParams["font.family"] = "sans-serif"
#rcParams["font.sans-serif"] = ["Computer Modern Sans"]
#rcParams["text.usetex"] = True
#rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"


all_model_names = ['LR','LDA','KNN','CART','NB','SVM','MVH','MVS']
#warnings.filterwarnings("ignore", category=FutureWarning)
DEFAULT_INPUT_ML_DATA_PATH = 'data/ML_data_cloud_rayleigh/ML_'
special_read_list = {'agrococcus':'spectra/Hegde_2013/Agrococcus/Agrococcus%20sp._KM349956.csv',
                     'geodermatophilus': 'spectra/Hegde_2013/Geodermatophilus/Geodermatophilus%20sp._KM349882.csv',
                     'bark': 'spectra/ECOSTRESS_nonphotosyntheticvegetation/bark/nonphotosyntheticvegetation.bark.acer.rubrum.vswir.acru-1-81.ucsb.asd.spectrum.csv',
                     'lichen': 'spectra/ECOSTRESS_nonphotosyntheticvegetation/lichen/nonphotosyntheticvegetation.lichen.lichen.species.vswir.vh298.ucsb.asd.spectrum.csv',
                     'aspen': 'spectra/USGS_Vegetation_and_Microorganisms/Aspen_Leaf-A/aspen_leaf_dw92-2.10753.asc',
                     'leafy spurge': 'spectra/USGS_Vegetation_and_Microorganisms/LeafySpurge/leafyspurge_spurge-a2-jun98.11306.asc'
                    }

################################# PARAMETERS ##################################

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode',  type=str, default=None, required=True,
                    choices = ['plot', 'full'], help='Plot or full?' )
parser.add_argument('-t', '--ttype', type=str, required=True,
                    choices = ['binary', 'multiclass'],
                    help='Training types: Multiclass or binary training')
parser.add_argument('-mpc', '--cores',  type=int, default=10, required=True,
                    help='How many cores for multiprocessing?' )
parser.add_argument('-snr', '--snr', type=int, default=None, required=True,
                    help='SNR to train' )

parser.add_argument('-sc', '--scaler', type=str, default='default',
                    help='Path to the scaler')
parser.add_argument('-o', '--output', type=str,
                    default = 'default',
                    help='Output directory' )
parser.add_argument('-k', '--kfold', type=int, default=10,
                    help='K in k-fold' )
parser.add_argument('-s', '--seed', type=int, default=10,
                help='Seed for the random state. Set to None for full random' )
parser.add_argument('-ks', '--kshuffle', type=bool, default=True,
                    help='Shuffle while doing K-fold?' )
args = parser.parse_args()

mode = args.mode
training_mode = args.ttype
snr = args.snr
kfold_K = args.kfold
cores = args.cores
seed  = args.seed

if args.output == 'default':
    if training_mode == 'binary':
        output_dir = 'output/binary_class'
    else:
        output_dir = 'output/multiclass'
else:
    output_dir = args.output

if args.output == 'default':
    if training_mode == 'binary':
        scaler_dir = 'output/binary_class/all/scaler.pk'
    else:
        scaler_dir = 'output/multiclass/all/scaler.pk'
else:
    scaler_dir = args.scaler

scaler = pk.load(open(scaler_dir, 'rb'))

result_input_data_file= f'{output_dir}/predictions_snr_{snr}_k_{kfold_K}.pk'
prob_input_data_file= f'{output_dir}/probabilities_snr_{snr}_k_{kfold_K}.pk'

time_of_run = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
initial_time = time.time()
np.random.seed(seed)

print("Using {} cores".format(cores))
print("Seed: {}".format(seed))
print("SNR: {}".format(snr))
print("K-fold K = {}".format(kfold_K))
print("Running on {} mode".format(mode))
print("Data scaler: {}".format(scaler_dir))
print("Run starts at {}\n".format(time_of_run))
style.use('seaborn-dark-palette')
###############################################################################

### HELPER FUNCTIONS FOR MULTIPROCESSING ###
def make_training_data_fixed_snr(category_name,snr,training_mode):
    """ Make training given a fixed SNR (e.g. snr=100).

        Parameters:
            category_name (str): name of the biota
            snr (float or int): the signal to noise ratio
            training_mode (str): either binary or multiclass
        
        Returns:
            training (array): training dataset 
            training_class (array): the validation dataset 
                (e.g. array of 0 or 1 with length = len(training) for binary mode)
            validation_veg_comp (array): array of vegetation percent
            category_name (str): name of the biota 

    """
    assert (type(snr) == float or type(snr) == int)
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
    if training_mode == 'binary':
        training_class = []
        for i in category_df['classification']:
            if i['seawater'] > 0:
                training_class.append(1) #true, vegetation
            elif i['seawater'] == 0:
                training_class.append(0) #false, no vegetation
            else:
                raise ValueError
    elif training_mode == 'multiclass':
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

    training_class = array(training_class)
    training = array(training)
    validation_veg_comp = array(validation_veg_comp)
    
    return training, training_class, validation_veg_comp, category_name

def predict_model(models, training, i, category_name):
    """
        Predicting on the *trained* models (use the all_veg_training.py first).
        Returns the predictions, predictions probability. Note that the MVH
        method cannot return prediction probability due its nature of unanimous
        voting.

        Parameters:
            models (list): trained models
            training (array): training dataset
            i (int): which k-fold were these models trained on 
                    (for housekeeping purpose)
            category_name (str): the biota name (for housekeeping purpose)

        Returns:
            predictions (dict): the predictions based on the training array.
                Shape = key: (length of training array,)
            predictions_probability (dict): the probability of predictions.
                Shape = (length of predictions array,)
            i (array): same i as the parameters
            category_name (str): same category_name as the parameters
            all_proba (dict): the probability of predicting each of classes.
                Shape is key: (length of training array, number of classes)

    """

    print('Predicting on k={} and biota {}'.format(i+1, category_name))
    
    predictions = {}
    predictions_probability = {} #store average probability of prediction
    all_proba = {}
    
    for name, model in models:
        
        #predict
        pred = model.predict(training)
        predictions[name] = pred
        
        if name != 'MVH':
            fullproba = model.predict_proba(training) #prediction probability
            # note: fullproba is an array of length d, where d is the number
            # of classes (e.g. d = 2 for binary classification)
            
            predicted_proba = []
            for j in range(len(fullproba)):
                #get the probability of the prediction
                class_index = list(model.classes_).index(pred[j])
                predicted_proba.append(fullproba[j][class_index])
            
            predictions_probability[name] = array(predicted_proba)
            all_proba[name] = fullproba
            
        elif name == 'MVH': #MVH cannot do probability!
            
            #return an array of np.NaN instead (the heatmap can handle this!)
            empty_nan = np.empty(shape=len(pred))
            empty_nan[:] = np.nan
            
            #return an array of np.NaN also
            fullproba = np.empty(shape=(len(pred),
                                len( list(model.classes_) ) ) 
                                )
            fullproba[:] = np.nan

            predictions_probability[name] = empty_nan
            all_proba[name] = fullproba

    return predictions, predictions_probability, all_proba, i, category_name

def __index_first_non_zero(val):
    """ Helper function for prep_label(). Find the first non-zero, non-decimal
        index of a number val.

        Parameters:
            val (float or int): the value that we want to get the index

        Returns:
            i: the index
    """

    val = str(val)
    for i in range(len(val)):
        if val[i] != '0' and val[i] != '.':
            return i

def prep_label(std, val):
    """ Make the label for the heatmap. Takes in the mean and std of a number
        and return a string of the form number(std). 
        
        e.g. 0.05 +/- 0.004 --> 0.05(4)
        
        Parameters:
            std (float): the standard deviation
            val (float): the mean (the value)

        Returns:
            text_label (str): the string formatted in the form val(std)
    """
    
    std = '{:.123f}'.format(std) #force all floats to decimal notation
    #there are special cases where the std == 0.0 
    #(LR/LDA for agrococcus)
    if std == 0.0:
        text_label = val
        return text_label
    
    if __index_first_non_zero(std) != None:
        std_exp = __index_first_non_zero(std) - 1
        std_val = str(round(float(std), std_exp))[-1]

    #prepare the label in the form average(std)
    text_label = (str(round(val, 
                        std_exp)) + 
                '({})'.format(std_val))
    rounding = '{:.' + str(std_exp-1) + 'f}'

    text_label = rounding.format(val) + '({})'.format(std_val)

    return text_label

def make_heatmap(heatmap, heatmap_label, ax, fig, 
                aspect=1, colorbar=False,
                cmap = cm.YlGnBu_r, cmap_bad = '#bdbdbd'):
    """ Make a heatmap from array heatmap, and label it with the array
        heatmap_label.

        Parameters:
            heatmap (array): value for the color of the heatmap
            heatmap_label (array): the label for each cell.
            ax (obj): axes plotting object from plt.subplots
            fig (obj): fig plotting object from plt.subplots
            aspect (int, optional): the aspect ratio. Default: 1
            colorbar (bool, optional): do you want the colorbar? 
                Default = False
            cmap (obj): matplotlib's colormap object
            cmap_bad (str): color for cells with value np.NaN
        
        Returns:
            ax (obj): the axes plotting object

    """

    cmap.set_bad(color=cmap_bad)
    im = ax.imshow(heatmap,cmap = cmap, aspect=aspect,vmin=0,vmax=1)
    
    if colorbar:
        cbar = fig.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Average prediction probability', rotation=270)

    #ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('1')
    
     # Create text annotations.
    for i in range(len(heatmap_label)):
        for j in range(len(heatmap_label[i])):

            text = ax.text(j, i, heatmap_label[i][j][0],
                ha="center", va="center", color=heatmap_label[i][j][1], 
                fontsize = 7)
    
    ax.set_yticks(arange(0,len(all_model_names),1))
    ax.set_yticklabels(all_model_names, va='center')
    ax.yaxis.set_ticks_position('none') 

    ax.set_xticks(arange(0,len(all_X_val.keys()),1))
    xticklabels = [veg_name.capitalize() for veg_name in all_X_val.keys()]
    xticklabels[xticklabels.index('Geodermatophilus')] = 'Geoder-\nmatophilus'

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(all_X_val.keys()), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(all_model_names), 1), minor=True)
    ax.set_xticklabels(xticklabels,ha='right')
    ax.xaxis.set_ticks_position('none') 

    # Rotate the tick labels 
    setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    ax.grid(which='minor',color='w', linewidth=1)
    
    return ax

if __name__ == '__main__':
    np.random.seed(seed) #set the seed

    if mode == 'full':
        print("Running trained models on noise data with specific SNR")

        ### Make Training Data ###

        # I. make training data
        print('Making training data')

        #make training data
        training_data = make_training_data_snr_range('leafy spurge', SNR_min, 
                                                        SNR_max, training_mode)
    
        all_training, all_training_class, all_validation_veg = training_data
            
        #scale the training data
        all_training = scaler.transform(all_training)

        #make them np array
        all_training = array(all_training)
        all_training_class = array(all_training_class)
        all_validation_veg = array(all_validation_veg)
        
        all_X_val['leafy spurge'] = pd.DataFrame(training,columns=['B','V','R','I'])
        #true value
        all_X_val['validation'] = training_class
        #water %
        all_X_val['seawater'] = validation_veg_comp

        #initiliaze result holders
        #access both result_df and prob_df by result_df[k number][category_name][model name]
        result_df = {} #prediction results for training data
        prob_df   = {} #to hold probabilities (lots of data)

        tmp_df = {'leafy spurge': pd.DataFrame()} #template holder for prob_df

        #for each i, we want a template holder
        for i in range(kfold_K):
            result_df[i] = deepcopy(all_X_val) #copy the validation/training data
            prob_df[i]   = deepcopy(tmp_df)    #copy the template holder for prob_df

       
        print('Predicting on noisy data (t={})'.format(time.time() - initial_time))

        ### PREDICT ON TRAINING DATA ###

        #things to iterate over parallel processing
        args_for_mp = []
        for i in range(kfold_K):
            training = all_X_val['leafy spurge'][['B','V','R','I']]
            models = pk.load(open(f'{output_dir}/all/models_{i+1}.pk','rb'))
            args_for_mp.append((models, training, i, 'leafy spurge'))

        #parallel processing
        all_predictions = Parallel(n_jobs=cores, verbose=10)(delayed(
            predict_model)(models, training,i,category_name) 
                for models, training,i,category_name in args_for_mp )

        #consolidate paralleling processing results
        for prediction in all_predictions:

            prediction_result, prediction_probability, allproba, i, category_name = prediction

            #the predictions
            for model_name, model_result in prediction_result.items():
                result_df[i][category_name][model_name] = model_result
            
            #predprob = probability of the returned prediction
            for model_name, prob_result in prediction_probability.items():
                prob_df[i][category_name][f'{model_name}_predprob'] = prob_result
            
            #allprob = probabilities of ALL prediction classes
            for model_name, fullproba in allproba.items():
                for j in range(len(fullproba[0])):
                    prob_df[i][category_name][f'{model_name}_prob{j}'] = fullproba[:,j]
            prob_df[i][category_name]['classes'] = len(fullproba[0])
            

        ### SAVE DATA ###
        result_filepath = f'{output_dir}/predictions_snr_{snr}_k_{kfold_K}.pk'
        prob_filepath   = f'{output_dir}/probabilities_snr_{snr}_k_{kfold_K}.pk'

        pk.dump(result_df, open(result_filepath,'wb'))
        pk.dump(prob_df, open(prob_filepath, 'wb'))

    ### MAKE PLOTS ###

    all_ba_heatmaps = []
    all_prob_heatmaps = []
    roc_data = {} #roc curves
    conf_matrix_data = {} #confusion_matrix

    if mode == "plot":

        try:
            result_input_file = open(result_input_data_file, 'rb')
            prob_input_file = open(prob_input_data_file, 'rb')
        except FileNotFoundError:
            raise ValueError('Need to run on "full" mode first!')    

        result_df = pk.load(result_input_file)
        prob_df   = pk.load(prob_input_file)

    for i in range(kfold_K):
        print(f'Processing k={i+1}/{kfold_K}')
        all_X_val = result_df[i]

        heatmap = [] #ba heatmap
        prob_heatmap = []
        
        j = 0
        for veg in ['all'] + list(special_read_list.keys()):
            heatmap.append([])
            prob_heatmap.append([])
            
            if i ==0 : 
                roc_data[veg] = []
                conf_matrix_data[veg] = []

            for model_name in all_model_names:

                y_true = all_X_val[veg]['validation']
                y_pred = all_X_val[veg][model_name]
                y_predprob = prob_df[i][veg][model_name+"_predprob"]
                #y_predprob = all_X_val[veg][model_name+'_probability']
                
                classes_count = prob_df[i][veg]['classes'][0]
                probabilities = [prob_df[i][veg][f'{model_name}_prob{k}'] for 
                                    k in range(classes_count)]
                y_allprobs = array(probabilities).transpose()

                #HEATMAP
                heatmap[j].append(balanced_accuracy_score(y_true, y_pred))
                prob_heatmap[j].append(np.mean(y_predprob, axis=0))

                #confusion matrix
                if i == 0:
                    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
                    conf_matrix_data[veg].append([model_name, conf_matrix])

                #ROC CURVES FOR BINARY CLASSIFICATION
                if i == 0 and training_mode == 'binary' and model_name != 'MVH':
                    positive_probs = y_allprobs[:, 1]
                    fpr, tpr, thresholds = roc_curve(y_true, positive_probs)
                    #roc_auc = auc(y_true, y_pred)
                    #print(fpr, tpr)
                    roc_data[veg].append([model_name, fpr, tpr, thresholds])

                #HISTOGRAM FOR LEAFY SPURGE TO FIND OUT WHY
                binning = arange(0,105,5)
                if veg in ['leafy spurge'] and i == 0:
                    # get all incorrect/correct predictions
                    tmp_X_val = all_X_val[veg]
                    correct_df = (tmp_X_val.loc[tmp_X_val[model_name] == 
                        tmp_X_val['validation']]['vegetation'])
                    incorrect_df = (tmp_X_val.loc[tmp_X_val[model_name] != 
                        tmp_X_val['validation']]['vegetation'])
                    
                    # get the histogram
                    correct, bin_edges = np.histogram(correct_df,bins=binning)
                    incorrect, bin_edges = np.histogram(incorrect_df,bins=binning)
                    total = correct + incorrect

                    # plot the prediction ratios
                    figure()
                    bar(bin_edges[:-1], correct/total, width=5, label='Accurate', 
                        fill=False, hatch='/', linewidth=1.2, align = 'edge')                        
                    bar(bin_edges[:-1], incorrect/total, width=5, label='Inaccurate',
                        alpha=0.8, color='#d55c00', linewidth=1.2,  align='edge')

                    # details for the plots
                    title('{}, {}, SNR={}'.format(model_name, veg, snr))
                    xticks(arange(0, 105, 10))
                    xlim([0,100])                        
                    ylim([0,1])
                    ylabel('Prediction Ratio')
                    xlabel('Biota Percentage')
                    legend()
                    savefig(f'{output_dir}/all/{veg}_{model_name}_{snr}.pdf')
                        
            j+=1
        
        ba_heatmap = array(heatmap).transpose()        
        prob_heatmap = array(prob_heatmap).transpose()

        all_ba_heatmaps.append(ba_heatmap)
        all_prob_heatmaps.append(prob_heatmap)

    average_ba_heatmap = np.mean(all_ba_heatmaps,axis=0)
    std_ba_heatmap = 5*np.std(all_ba_heatmaps, axis=0) #report 5 sigmas
    average_prob_heatmap = np.mean(all_prob_heatmaps, axis=0)

    heatmap_labels =[]
    for i in range(len(average_ba_heatmap)):
        heatmap_labels.append([])
        for j in range(len(average_ba_heatmap[i])):
            if average_ba_heatmap[i,j]<0.4:
                text_color = 'w'
            else: 
                text_color='k'

            #prepare the label in the form average(std)
            text_label = prep_label(std_ba_heatmap[i,j], 
                                    average_ba_heatmap[i,j])
            
            heatmap_labels[i].append([text_label, text_color])

    fig,axes = subplots(nrows=1,ncols=1,figsize=(6,6))
    ax1 = make_heatmap(average_prob_heatmap, heatmap_labels, axes, fig, colorbar=True)

    subplots_adjust(wspace=0.05,bottom=0.15)
    fig.suptitle('SNR = {}'.format(snr))
    savefig(output_dir + '/best_scores_SNR_{}.pdf'.format(snr))
    close('all')

    for veg, all_roc_val in roc_data.items():
        figure()
        for roc_val in all_roc_val:
            model_name, fpr, tpr, thresholds = roc_val
            # Plot ROC curve
            plot(fpr, tpr, label=f'{model_name}')

        plot([0, 1], [0, 1], 'k--')  # random predictions curve
        xlim([0.0, 1.0])
        ylim([0.0, 1.0])
        xlabel('False Positive Rate or (1 - Specifity)')
        ylabel('True Positive Rate or (Sensitivity)')
        title(f'Receiver Operating Characteristic, {veg}, S/R {snr}')
        legend(loc="lower right")
        savefig(f'{output_dir}/all/ROC_{veg}_{snr}.pdf')
        #show()
    close('all')

    for veg, alldata in conf_matrix_data.items():
        fig, axes = subplots(nrows = 4, ncols = 2, sharex = True, sharey=True, figsize=(8,6))
        fig.subplots_adjust(wspace=0.02,hspace=0.35)
        axes = axes.ravel()
        j = 0
        for data in alldata:
            ax = axes[j]
            model_name, conf_matrix = data
            # Transform to df for easier plotting
            conf_matrix = pd.DataFrame(conf_matrix,
                                index = ['False', 'True'], 
                                columns = ['False', 'True'])
            
            ax = sns.heatmap(conf_matrix, annot=True, vmin=0, vmax=1, ax = ax, 
                             cbar=False, cmap=cm.YlGnBu_r)

            if j%2 == 0:  ax.set_ylabel('True label')
            ax.tick_params(right= False,top= False, bottom=False, left=False)
            if j in [6, 7]: ax.set_xlabel('Predicted label')
            ax.set_title(model_name, color='k')
            
            j+=1
        fig.suptitle(f'Normalized Confusion Matrix\n{veg.capitalize()}, S/R {snr}')
        savefig(f'{output_dir}/all/confusion_{veg}_{snr}.pdf')

        


