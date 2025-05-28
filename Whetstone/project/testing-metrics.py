
import warnings
import os
import sys
import re
warnings.filterwarnings("ignore")
#sys.stderr = open(os.devnull, 'w')

from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import distance
from sklearn.decomposition import PCA

# Turn on scikit-learn optimizations :
from sklearnex import patch_sklearn
patch_sklearn()

import aif360.datasets as datasets
import aif360.metrics as metrics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from main_transform import *
from var_utils import *

def fill_x(og_data, data):
        diff = list(set(og_data.columns)-set(data.columns))
        if len(diff) == 0:
                return data
        add_cols = pd.DataFrame(0, index=np.arange(len(data)), columns=diff)
        data= pd.concat([data,add_cols],axis=1)
        return data[list(og_data.columns)]

def calc_min_change_metrics(data, protected, predicted):
	privileged = 1
	unprivileged = 0
	preferred = 1
	unpreferred = 0
	priv_groups = [{protected: privileged}]
	unpriv_groups = [{protected: unprivileged}]
	if 'fnlwgt' in data.columns:
		weights_name = 'fnlwgt'
	else:
		weights_name = None

	## PERFECT ACCURACY
	dataset= datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
	no_pred_classMetrics = metrics.ClassificationMetric(dataset, dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups)
	nopred_DI =  1-min(no_pred_classMetrics.disparate_impact(), 1/(no_pred_classMetrics.disparate_impact()))
	print("IF PERFECT ACC - DISP= ", nopred_DI)
	## VALUE_COUNTS
	val_counts = data.value_counts([protected, predicted])
	print(val_counts)
	## PROBABILITIES
	s0 = data[data[protected] == 0]
	y1_given_s0 = s0[s0[predicted] == 1]
	prob_y1_given_s0 = len(y1_given_s0)/len(s0)
	s1 = data[data[protected] == 1]
	y1_given_s1 = s1[s1[predicted] == 1]
	prob_y1_given_s1 = len(y1_given_s1)/len(s1)
	y0 = len(data[data[predicted] == 0].index)
	y1 = len(data[data[predicted] == 1].index)
	print("s0: ", len(s0.index))
	print("s1: ", len(s1.index))
	print("y1|s0: ",len(y1_given_s0.index))
	print("y1|s1: ",len(y1_given_s1.index))
	print("prob(y1|s0): ",prob_y1_given_s0)
	print("prob(y1|s1): ",prob_y1_given_s1)
	## MINIMUM CHANGE
	s0_len = len(s0.index)
	s1_len = len(s1.index)
	y1_given_s0_len = len(y1_given_s0.index)
	y1_given_s1_len = len(y1_given_s1.index)
	A = y1_given_s0_len
	B = s0_len
	C = y1_given_s1_len
	D = s1_len	
	################ CHANGES TO GET PERFECT DI #####################
	print("################### DI = 1 #################")
	## IF ONLY CHANGING UNDERPRIVILEGED
	i_u = int(B*C/D - A)
	print("ONLY CHANGING UNDERPRIVILEGED: ",i_u)
	print("NEW DISP: ", ((A+i_u)/B)/(C/D))
	print("NEW ACC: ", (B+D-np.abs(i_u))/(B+D))
	print("NEW BAL ACC: ", .5*( y1/(y1+max(i_u,0))+ y0/(y0+max(-i_u,0))))
	## IF ONLY CHANGING PRIVILEGED
	i_p = int(A*D/B - C)
	print("ONLY CHANGING PRIVILEGED: ",i_p)
	print("NEW DISP: ", (A/B)/((C+i_p)/D))
	print("NEW ACC: ", (B+D-np.abs(i_p))/(B+D))
	print("NEW BAL ACC: ", .5*(y1/(y1+max(-i_p,0))+ y0/(y0+max(i_p,0))))
	## IF CHANGING HALF OF EACH
	print("HALF CHANGING UNDERPRIVILEGED: ",i_u/2)
	print("HALF CHANGING PRIVILEGED: ",i_p/2)
	print("TOTAL CHANGED: ",np.abs(i_p/2)+np.abs(i_u/2))
	print("NEW DISP: ", ((A+i_u/2)/B)/((C+i_p/2)/D))
	print("NEW ACC: ", (B+D-np.abs(i_p)-np.abs(i_u))/(B+D))
	print("NEW BAL ACC: ", .5*(y1/(y1+max(-i_p,0)+max(i_u,0))+ y0/(y0+max(i_p,0)+max(-i_u,0))))
	################ CHANGES TO GET LEGAN DI (.8) #####################
	print("################### DI = .8 #################")
	## IF ONLY CHANGING UNDERPRIVILEGED
	i_u = int(.8*B*C/D - A)
	print("ONLY CHANGING UNDERPRIVILEGED: ",i_u)
	print("NEW DISP: ", ((A+i_u)/B)/(C/D))
	print("NEW ACC: ", (B+D-np.abs(i_u))/(B+D))
	print("NEW BAL ACC: ", .5*( y1/(y1+max(i_u,0))+ y0/(y0+max(-i_u,0))))
	## IF ONLY CHANGING PRIVILEGED
	i_p = int((A*D/B - .8*C)/.8)
	print("ONLY CHANGING PRIVILEGED: ",i_p)
	print("NEW DISP: ", (A/B)/((C+i_p)/D))
	print("NEW ACC: ", (B+D-np.abs(i_p))/(B+D))
	print("NEW BAL ACC: ", .5*(y1/(y1+max(-i_p,0))+ y0/(y0+max(i_p,0))))
	## IF CHANGING HALF OF EACH
	print("HALF CHANGING UNDERPRIVILEGED: ",i_u/2)
	print("HALF CHANGING PRIVILEGED: ",i_p/2)
	print("TOTAL CHANGED: ",np.abs(i_p/2)+np.abs(i_u/2))
	print("NEW DISP: ", ((A+i_u/2)/B)/((C+i_p/2)/D))
	print("NEW ACC: ", (B+D-np.abs(i_p)-np.abs(i_u))/(B+D))
	print("NEW BAL ACC: ", .5*(y1/(y1+max(-i_p,0)+max(i_u,0))+ y0/(y0+max(i_p,0)+max(-i_u,0))))
	
	return

		


def calc_metrics(data, og_test_data, protected, predicted):
	alg = 'RandomForest'

	privileged = 1
	unprivileged = 0
	preferred = 1
	unpreferred = 0
	priv_groups = [{protected: privileged}]
	unpriv_groups = [{protected: unprivileged}]

	if 'fnlwgt' in og_test_data.columns:
		weights_name = 'fnlwgt'
	else:
		weights_name = None
	DI = 0
	stat_par = 0
	avg_odds = 0
	eq_op = 0
	theil = 0 

	bal_acc = 0
	prec = 0 
	rec = 0
	acc = 0
	avg = 3 
	if protected == 'gender' and predicted == 'labels':
		avg = 6
	for i in range(avg):
		###PREDICT OG DATA
		gen_reg = train_on_data(data)
		og_predicted_data = predict_output(og_test_data, gen_reg) #Predict original on generated trained classifier

		## ORGANIZE ORIGINAL DATA
		og_dataset= datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=og_test_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
		###ORGANIZE DATA TRAINED ON GENERATED AND TESTED ON ORIGINAL TEST DATA
		og_predicted_dataset = datasets.BinaryLabelDataset(favorable_label=preferred, unfavorable_label=unpreferred, df=og_predicted_data, label_names=[predicted], protected_attribute_names=[protected], instance_weights_name = weights_name)
		og_classMetrics = metrics.ClassificationMetric(og_dataset, og_predicted_dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups)# Predicted Original data on generated trained classifer
		
		## DISPARATE IMPACT
		DI_intermede =  1-min(og_classMetrics.disparate_impact(), 1/(og_classMetrics.disparate_impact()))
		DI = DI + DI_intermede
		## STATISTICAL PARITY
		stat_par = stat_par + og_classMetrics.statistical_parity_difference()
		## AVERAGE ODDS
		avg_odds = avg_odds + og_classMetrics.average_odds_difference()
		## EQUAL OPPORTUNITY
		eq_op = eq_op + og_classMetrics.equal_opportunity_difference()
		## THIEL
		theil = theil + og_classMetrics.theil_index()
		## BALANCED ACCURACY
		bal_acc = bal_acc + 0.5*(og_classMetrics.true_positive_rate()+og_classMetrics.true_negative_rate())
		## PRECISION
		prec =  prec + og_classMetrics.precision()
		## RECALL
		rec = rec + og_classMetrics.recall()
		## ACCURACY
		acc =  acc + og_classMetrics.accuracy()
	DI = DI/avg 
	avg_odds = avg_odds/avg
	eq_op = eq_op/avg
	theil = theil/avg
	bal_acc = bal_acc/avg 
	prec = prec/avg 
	rec = rec/avg 
	acc = acc/avg 
	return DI, stat_par, avg_odds, eq_op, theil, prec, rec, acc, bal_acc, og_predicted_data


def run(file, og_data, og_file, test_data, protected, privileged, predicted, preferred, basic, all=False):
	global og_dist

	if not all:
		samples,_,_ = get_data(file, og_file, protected, privileged, predicted, preferred)
		samples = fill_x(og_data, samples)
	else:
		samples = file
		file = "All_files.csv"

	DI, stat_par, avg_odds, eq_op, theil, prec, rec, acc, bal_acc, predicted_data = calc_metrics(samples, test_data, protected, predicted)
	DI = DI.round(3)
	stat_par = stat_par.round(3)
	avg_odds = avg_odds.round(3)
	eq_op = eq_op.round(3)
	theil = theil.round(3)
	prec = prec.round(3)
	rec = rec.round(3)
	acc = acc.round(3)
	bal_acc = bal_acc.round(3)
	print("Disp Imp: ",DI, " Precision: ",prec," Recall: ",rec," Accuracy: ",acc, " Bal Acc: ", bal_acc)
	if file == og_file:
		file = 'Original'
	if basic and basic in file: 
		file = 'Base'

	##GET DIVERSITY OF DATASET
	from sklearn.decomposition import PCA

	# Assuming 'data' is your dataset
	pca = PCA(n_components=1)
	pca.fit(samples)
	pca_data = pca.fit_transform(samples)

	# Explained variance ratio gives the proportion of variance explained by each component
	explained_variance_ratio = pca.explained_variance_ratio_
	#print("Explained variance ratio by each principal component:", explained_variance_ratio)
	#print(sum(explained_variance_ratio))
	variances = np.var(pca_data, axis=0)
	#print("Variances for each feature:", variances)

	return [file.split('/')[-1], DI, stat_par, avg_odds, eq_op, theil, prec, rec, acc, bal_acc], samples
	
    
def get_og_file(input_file, protected):
    if 'adult' in input_file:
        og_file = 'real_data/adult/ADULT-SPLIT-TRAIN-60.csv'
        og_test = 'real_data/adult/ADULT-SPLIT-TEST-20.csv'
        #basic = 'ablation_study/ablation_study_data/ablation_study_GAN_type/adult-VGAN-norm_selected_samples/'
        basic = 'ablation_study/ablation_study_data/ablation_study_GAN_type/adult-WGAN-gmm_selected_samples/'
    elif 'compas' in input_file:
        og_test = 'real_data/propublica-compas/PROPUBLICA-COMPAS-SPLIT-TEST-20.csv'
        og_file = 'real_data/propublica-compas/PROPUBLICA-COMPAS-SPLIT-TRAIN-60.csv'
        basic = 'ablation_study/ablation_study_data/ablation_study_GAN_type/compas-'+protected+'-VGAN-norm_selected_samples/'
    elif 'german' in input_file:
        og_file = 'real_data/german/GERMAN-SPLIT-TRAIN-60.csv'
        og_test = 'real_data/german/GERMAN-SPLIT-TEST-20.csv'
        basic = 'ablation_study/ablation_study_data/ablation_study_GAN_type/german-VGAN-norm_selected_samples/'
    elif 'bank' in input_file:
        og_file = 'real_data/bank/bank-full-SPLIT-TRAIN-60-age.csv'
        og_test = 'real_data/bank/bank-full-SPLIT-TEST-20-age.csv'
        basic = 'ablation_study/ablation_study_data/ablation_study_GAN_type/bank-VGAN-norm_selected_samples/'
    elif 'medical' in input_file:
        og_file = 'real_data/medical/meps21-SPLIT-TRAIN-60.csv'
        og_test = 'real_data/medical/meps21-SPLIT-TEST-20.csv'
        basic = 'ablation_study/ablation_study_data/ablation_study_GAN_type/medical-VGAN-norm_selected_samples/'
    return og_file, og_test, basic

if __name__ == "__main__":
	
	dataset = sys.argv[1]
	protected = sys.argv[2] #Test Protected Value
	if len(sys.argv) > 3: protected_select = sys.argv[3] #Selected for Protected Value
	else: protected_select = protected

	dataset_name = 'unknown'
	basic = False 
	data_name = [ x for x in ['adult','bank','compas','german','medical'] if x in dataset.lower() ]
	if len(data_name) > 0: dataset_name = data_name[0]
	og_file, test_file, basic = get_og_file(dataset_name, protected_select)
		
	print("Original: ",og_file," Test: ",test_file, " Basic: ",basic)
	if len(sys.argv) > 4:
        ## Get Ablation Test Data Files
		ablation_folder = sys.argv[4]
		data_folders = [ f'{ablation_folder}/{folder}' for folder in os.listdir(ablation_folder) if os.path.isdir(os.path.join(ablation_folder,folder)) and dataset_name in f'{ablation_folder}/{folder}' ] 
		data_files =  [ f'{ablation_folder}/{file}' for file in os.listdir(ablation_folder) if file.startswith('sample_data') and dataset_name in f'{ablation_folder}/{file}' ] 
		for testing_folder in data_folders: data_files = data_files + [ f'{testing_folder}/{file}' for file in os.listdir(testing_folder) if file.startswith('sample_data') ] 
        
		if 'NAIVE2' in ablation_folder and dataset_name != 'adult': data_files = [x for x in data_files if 'VGAN-norm' in x]
		if 'NAIVE2' in ablation_folder and dataset_name == 'adult': data_files = [x for x in data_files if 'WGAN-gmm' in x]
		if 'disp_select_all' in ablation_folder or 'gan_type' in ablation_folder.lower(): basic = False
		else: data_files = data_files + [basic+'/'+file for file in os.listdir(basic) if file.startswith('sample_data')]
        
        ## Ignore Alternative Protected Files
		if protected_select == 'gender': data_files = [x for x in data_files if not 'race' in x]
		elif protected_select == 'race': data_files = [x for x in data_files if not 'gender' in x and not 'sex' in x]
	else:
        ## Get Base Data Files
		data_files = [basic+'/'+file for file in os.listdir(basic) if file.startswith('sample_data')]
	data_files = [og_file] + sorted(list(set(data_files)))
	protected, privileged, _, predicted, preferred, _ = get_labels(og_file, protected)
	og_data,_,_ = get_data(og_file, og_file, protected, privileged, predicted, preferred)
	test_data,_,_ = get_data(test_file, og_file, protected, privileged, predicted, preferred)
	test_data = fill_x(og_data, test_data)
	all_data = pd.DataFrame(columns=og_data.columns)
	#calc_min_change_metrics(test_data, protected, predicted)

    ## Create Folder For Saving Metrics
	mets_folder = f'mode_collapse/{dataset}/'
	if protected != protected_select: mets_folder = f'mode_collapse/{dataset}_{protected}/'
	if not os.path.exists(mets_folder): os.makedirs(mets_folder)
	print(mets_folder)

    ## Read or Create File for Saving Metrics
	metric_columns=['file','DI', 'Stat Par', 'Avg Odds','Eq Opp','Theil','prec','rec','acc', 'bal_acc']
	try:
		mets_pd = pd.read_csv(f'{mets_folder}/metrics.csv',index_col=0)
		if list(mets_pd.columns) != metric_columns: mets_pd = pd.DataFrame(columns=metric_columns)
	except:
		mets_pd = pd.DataFrame(columns=metric_columns)
	done_files = [file for file in mets_pd['file']]

    ## Test all Data Files
	for data_file in data_files:
		print(data_file)
		file_name = data_file
		if file_name == og_file: file_name = 'Original' 
		if basic != False:
			if file_name in [basic+'/'+file for file in os.listdir(basic) if file.startswith('sample_data')]: file_name = 'Base'
		if file_name.split('/')[-1] not in done_files:
			m, data = run(data_file, og_data, og_file, test_data, protected, privileged, predicted, preferred, basic)
			mets_pd.loc[len(mets_pd.index)] = m
			mets_pd.to_csv(f'{mets_folder}/metrics.csv')
	#for col in mets_pd.columns:
		#if col != "file":
			#mets_pd[col] = mets_pd[col].abs()
	mets_pd['config'] = [ re.sub(r'[\d_]+(-age)?.csv$', '', x) for x in mets_pd['file'] ] 
	mets_pd_plot = mets_pd.copy(deep=True)
	mets_pd_plot.index = [ re.search(r'\.?[\d,*]+',val).group() if re.search(r'\d+',val) else val for val in mets_pd_plot['config'] ]
	mets_pd_plot.to_csv(f'{mets_folder}/metrics-plot-all.csv')
	mets_avg = mets_pd.drop(columns=['file']).groupby(['config']).mean()
	#mets_avg.index = [ re.search(r'\.?[\d,*]+',val).group() if re.search(r'\d+',val) else val for val in mets_avg.index ]
	mets_avg.index = [ val.replace('sample_data_','').replace('parallel_','').replace(dataset_name+'-','')for val in mets_avg.index ]
	mets_std_err = mets_pd.drop(columns=['file']).groupby(['config']).sem()
	mets_std_err.columns = [ col+'_sem' for col in mets_std_err.columns ]
	#mets_std_err.index = [ re.search(r'\.?[\d,*]+',val).group() if re.search(r'\d+',val) else val for val in mets_std_err.index ]
	mets_std_err.index = [ val.replace('sample_data_','').replace('parallel_','').replace(dataset_name+'-','') for val in mets_std_err.index ]
	mets_avg = pd.concat([mets_avg, mets_std_err], axis=1)
	mets_avg.sort_index(inplace=True)
	mets_avg.to_csv(f'{mets_folder}/metrics-averages.csv')
	print(mets_avg)
print("END TESTING METRICS")
