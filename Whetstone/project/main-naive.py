import sys
import os
import warnings
import multiprocessing as mp 
import time
import math
import random
from operator import add
from sklearn.decomposition import PCA
from scipy.spatial import distance
import aif360.metrics as metrics
import aif360.datasets as datasets

# Turn on scikit-learn optimizations :
from sklearnex import patch_sklearn
patch_sklearn()

from main_transform import * 
from sklearnex.model_selection import train_test_split



def parallel_test(x):
	data = x[0]
	test_data = x[1].reset_index(drop=True)
	avg = x[2] 
	val_counts = data.value_counts([protected,predicted])
	for i in range(avg):
		reg = train_on_data(data)
		pred = predict_output(test_data, reg)
		out = call_calc(test_data, pred)
		if i == 0:
			out_avg = out	
		else:
			out_avg = [ x+y for x,y in zip(out_avg, out)]
	out = [ x/avg for x in out_avg ]
	out_nopred = call_calc(data)
	return [reg, pred, out, out_nopred]
		
def get_train_test(data, split=True):
	og_x = data[data.columns[:-1]].astype(float)
	og_y = data[data.columns[-1]].astype(float)
	if not split:
		return og_x, og_y
	x_train, x_test, y_train, y_test = train_test_split(og_x, og_y, test_size=0.33, stratify=og_y)
	return x_train, x_test, y_train, y_test

def train_on_data(data, return_test=False, return_train=False, split=False):
	#reg = RandomForestClassifier(n_estimators=500) 
	if use_gpu:
		reg = RandomForestClassifier(n_estimators=100) 
	else:
		reg = RandomForestClassifier(n_estimators=100, n_jobs=-1) 

	if split:
		x_train, x_test, y_train, y_test = get_train_test(data, split)
	else:
		x_train, y_train = get_train_test(data, split)


	if 'fnlwgt' in data.columns:
		weight = x_train['fnlwgt']
		x_train_with_weight = x_train.copy(deep=True)
		x_train = x_train.drop(columns=['fnlwgt'])
	else:
		weight = None	


	if use_gpu:
		if type(x_train) != type(cudf.from_pandas(pd.DataFrame(columns=[1]))):
			x_train_cudf = cudf.from_pandas(x_train)
			y_train_cudf = cudf.from_pandas(y_train)

	try:
		if use_gpu:
			reg.fit(x_train_cudf, y_train_cudf)#, sample_weight = weight)
		else:
			reg.fit(x_train,y_train, sample_weight = weight)
	except Exception as e:
		print("FAILED TRAINING: ",e)
		reg = -1

	if 'fnlwgt' in data.columns:
		x_train = x_train_with_weight
	if return_test:
		if use_gpu:
			test = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True).rename(og_data.columns[-1])],axis=1).reset_index(drop=True)
		else:
			test = pd.concat([x_test,y_test],axis=1).reset_index(drop=True)
		if return_train:
			if use_gpu:
				train = pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(og_data.columns[-1])],axis=1).reset_index(drop=True)
			else:
				train = pd.concat([x_train,y_train],axis=1).reset_index(drop=True)
			return reg, test, train
		return reg, test 
	return reg

def predict_output(data, reg=None):
	if isinstance(data, tuple):
		reg = data[1]
		data = data[0]
	x = data[data.columns[:-1]]
	if use_gpu:
		x.reset_index(drop=True)
	else:
		x = pd.DataFrame(x).reset_index(drop=True)
	if 'fnlwgt' in x.columns:
		x_with_weight = x.copy(deep=True)
		x = x.drop(columns = ['fnlwgt'])
	if reg == -1:
		y_pred = np.zeros([len(data)])
		print('ZEROS')
	else:
		try:
			y_pred = reg.predict(x)
		except:
			y_pred = np.zeros([len(data)])
			print('ZEROS')
	if 'fnlwgt' in data.columns:
		x = x_with_weight
	return pd.concat([x,pd.Series(y_pred,name=data.columns[-1])], axis=1)


def get_start_sample(data, start_sample_size, predicted, stratified=''):
    cols = list(data.columns)
    cols.remove(predicted)
    og_x = data[cols].astype(float)
    og_y = data[predicted].astype(float)
    if stratified != '':
        x_train, x_test, y_train, y_test = train_test_split(og_x, og_y, test_size=start_sample_size, stratify=data[stratified], shuffle=True)
    else:
        x_train, x_test, y_train, y_test = train_test_split(og_x, og_y, test_size=start_sample_size, shuffle=True)
    return pd.concat([x_train,y_train], axis=1), pd.concat([x_test, y_test], axis=1)

def fill_x(og_data, data):
    diff = list(set(og_data.columns)-set(data.columns))
    add_cols = pd.DataFrame(0, index=np.arange(len(data)), columns=diff)
    data = pd.concat([data,add_cols],axis=1)
    return data[og_data.columns]

def call_calc(new,predicted_data=''):
    fno = False
    if type(predicted_data) == type(''):
        predicted_data = new
        fno = True

    priv_groups = [{protected: 1}]
    unpriv_groups = [{protected: 0}]

    ## ORGANIZE ORIGINAL DATA
    og_dataset= datasets.BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=new, label_names=[predicted], protected_attribute_names=[protected])
    ###ORGANIZE DATA TRAINED ON GENERATED AND TESTED ON ORIGINAL TEST DATA
    og_predicted_dataset = datasets.BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=predicted_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = None)
    og_classMetrics = metrics.ClassificationMetric(og_dataset, og_predicted_dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups)# Predicted Original data on generated trained classifer


    f = min(og_classMetrics.disparate_impact(), 1/(og_classMetrics.disparate_impact()))
    if fno:
        return f

    a = 0.5*(og_classMetrics.true_positive_rate()+og_classMetrics.true_negative_rate())

    return [f, a]

    
def train(directory, protected, labels, alpha, to_select, beta, set_perc, hyp_disp, min_acc_bound, avail_num, try_num, ablation_var ):
    ## GET ORIGINAL VARIABLES
    train_start = time.time()
    files = [f for f in os.listdir(directory) if f.startswith('sample_data_')]
    print(files)
    file_num = len(files)
    priv = 1
    unpriv = 0
    pref = 1
    unpref = 0 
    
    ######## START WITH RANDOM DATA #######
    i = 0
    for file in files:
        data,_,_ = get_data(directory+"/"+file, og_file, protected, privileged, predicted, preferred)
        data = fill_x(og_data, data) 
        
        ## Combine all data into one dataset
        if i == 0:
            all_data = data
        else:
            all_data = pd.concat([all_data, data], axis=0, ignore_index=True) 
        i = i + 1
    all_data = all_data.astype(float).dropna(axis='index').reset_index(drop=True)
    N = int(len(all_data.index)/len(files)) 
    val_counts = all_data.value_counts([protected,predicted])
    

    to_select = N 
    set_num = N
    print("FINAL DATSET SIZE: ",N)
    print("NUMBER OF POINTS TO CHOOSE: ",to_select)
    print("SET SIZE: ",set_num)
    avg=1

    all_data = all_data.reset_index(drop=True)
    test_data = og_data
    all_data = all_data.sample(frac=1).reset_index(drop=True)
    ######## END START WITH RANDOM SAMPLE ########
   

    #### INITIALIZE VARIABLES ####
    
    ## SELECT
    set_size = N 
    try_data_list = []
    for j in range(int(try_num)):
        try_data = all_data.sample(set_size) 
        try_data_list = try_data_list + [try_data] 
    ## TEST NEIGHBORS
    with mp.Pool() as pool:
        pool_res = pool.map(parallel_test, [(x, test_data, avg) for x in try_data_list])
    pool_reg, pool_predicted, pool_out, pool_out_nopred = map(list, zip(*pool_res))
    floss, aloss = map(list, zip(*pool_out))

    # Mask nan values
    floss_nan = np.isnan(floss)
    aloss_nan = np.isnan(aloss)
    mask = np.logical_or(floss_nan, aloss_nan)
    disp= np.ma.masked_array(floss, mask=mask)
    bal = np.ma.masked_array(aloss, mask=mask)  
    
    ## CHOOSE BEST NEIGHBOR
    disp_try = disp

    res = disp_try + bal
    best_indices = np.argsort( -res )[:len(labels)] 
    print(res)
    print(best_indices)

    final_directory = 'ablation_study/ablation_study_data/ablation_study_'+ablation_var+'/'+data_name+'_selected_samples'
    if not os.path.isdir(final_directory):
        os.makedirs(final_directory) 

    i = 0
    for label in labels:
        best_ind = best_indices[i]
        best_res = res[best_ind]
          
        # UPDATE DATASET
        try_data = try_data_list[best_ind]	
        new_dataset = try_data
 
        file_name = data_name+'_'+protected+'_'+str(label)
        final_file_name = final_directory+'/sample_data_naive2_'+file_name 
        save_dataset(new_dataset.copy(), final_file_name, protected, ohe, og_file)
        i = i + 1

def save_dataset(new_dataset_save, file_name, protected, ohe, og_file):
    ## Get Final Dataset
    new_dataset_save = new_dataset_save.reset_index(drop=True)
    new_dataset_save = return_data(new_dataset_save, ohe, og_file, protected, predicted, privileged, preferred)
    new_dataset_save.to_csv(file_name+'.csv', index=False)	
    return 

#################################################### MAIN ######################################################
if __name__ == "__main__":
    ############################
    # Get Data + One Hot Encode
    ############################
    global gen_reg
    use_gpu = False
    #use_gpu = True
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_columns',None)
    try:
        directory = sys.argv[1]
        print('Synthetic Data Directory: ',directory)
        data_name = sys.argv[2]
        print('Dataset Name: ',data_name)
        training_file = sys.argv[3]
        print('Training Dataset File: ',training_file)
        protected = sys.argv[4]
        print('Protected Feature: ',protected)
        privileged = sys.argv[5]
        print('Privileged Value: ',privileged)
        predicted = sys.argv[6]
        print('Predicted Feature: ',predicted)
        preferred = sys.argv[7]
        print('Preferred Value: ',preferred)


        # Get all Files
        og_file = training_file
        og_data,_,ohe = get_data(og_file, og_file, protected, privileged, predicted, preferred)


        select_percent = float(1)
	## Alpha = 1
        beta = 0 
	## One Selection
        set_perc = 100
        hyp_disp = "disp"
        min_acc_bound = 0 
        avail_num = "all"
	## Try A Lot
        try_num = 1000 
        
        ablation_var = "NAIVE2"

        print('Ablation: ',ablation_var)

        if use_gpu:
            from cuml.ensemble import RandomForestClassifier
            import cudf
            os.environ['CUDA_VIDIBLE_DEVISES']="0,1"
        else:
            from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        print(e)
        print('Please supply the following arguments: [directory] [dataset_name] [training_file] [protected feature] [privileged value] [predicted feature] [preferred value] [flag] [beta/set_perc/selected_data_percentage (ex. .5)] ]')  
        exit()
    
    alpha = 1

    labels = ['1','2','3','4','5','6']
    train(directory, protected, labels, alpha, select_percent, beta, set_perc, hyp_disp, min_acc_bound, avail_num, try_num, ablation_var)
