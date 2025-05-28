
import warnings
import os
import sys
import re
import random
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


def run(file, og_file, og_data, protected, privileged, predicted, preferred, set_name, dataset, all=False):
	global og_kmeans_centers
	global og_kmeans

	if not all:
		samples,_,_ = get_data(file, og_file, protected, privileged, predicted, preferred)
		samples = fill_x(og_data, samples)
	else:
		samples = file
		file = "All_files.csv"

	# Most Similar original datapoint
	this_kmeans = og_kmeans.predict(samples)

	return [set_name, len(np.unique(this_kmeans))], samples

def run_og(og_data):
	global og_kmeans_centers
	global og_kmeans

	og_kmeans = KMeans(n_clusters=int(len(og_data.index)/5))
	og_kmeans = og_kmeans.fit(og_data)
	og_kmeans_centers = og_kmeans.cluster_centers_
	this_kmeans = og_kmeans.predict(og_data)

	return ['ORIGINAL', len(np.unique(this_kmeans))] 

def reorganize_averages(df):
	df = df.T
	prefix = 'POST-FAIRGAN-'

	# Extract base keys
	drop_cols = [x for x in df.columns if '_all' in x and prefix in x]
	df = df.drop(columns=drop_cols)
	base_keys = df.columns.str.split(prefix).str[-1]
	base_keys = [ x.split('_all')[0] for x in base_keys ]

	# Group by base keys and concatenate values
	select_data = {}
	all_data = {}
	suffix = '_all'
	for base_key in np.unique(base_keys):
		select_values = [ df[f"{prefix}{base_key}"].values[0] if prefix+base_key in df.columns else 'N/A' ]
		select_data[base_key] = select_values
		all_values = [ df[f"{base_key}{suffix}"].values[0] if base_key+suffix in df.columns else 'N/A' ]
		all_data[base_key] = all_values
	
	# Create DataFrame from the new data
	base_keys = np.unique(base_keys)
	select_df = pd.DataFrame.from_dict(select_data)
	all_df = pd.DataFrame.from_dict(all_data)
	new_df = pd.concat([df, all_df, select_df], axis=0)
	new_df = new_df[base_keys]
	new_df = new_df.set_index(pd.Index(['Avg Before Select','All','Avg After Select']))
	new_df = new_df.sort_index(axis=1)

	# Display the new DataFrame
	return new_df


def get_modes(mets_pd, dataset, protected, set_name, data_files):
	'''
	if len(data_files) >= 5:
		data_files = random.sample(data_files, 5)
	else:
		data_files = random.choices(data_files, k=5)
	'''
	all_data = pd.DataFrame(columns=og_data.columns)
	all_data_returned = pd.DataFrame(columns=og_data_returned.columns)
	if set_name not in mets_pd['file'].unique():
		for data_file in data_files:
			print(data_file)
			m, samples  = run(data_file, og_file, og_data, protected, privileged, predicted, preferred, set_name, dataset)
			mets_pd.loc[len(mets_pd.index)] = m
			mets_pd.to_csv('mode_collapse/'+dataset+'/metrics.csv')
			all_data = pd.concat([all_data, samples])
			samples_returned = return_data(samples, ohe_og, og_file, protected, predicted, privileged, preferred) 
			all_data_returned = pd.concat([all_data_returned, samples_returned])
		if len(data_files) > 0:
			m, _ = run(all_data, og_data, og_file, protected, privileged, predicted, preferred, set_name+"_all", dataset, True)
			print(set_name+"_all")
			print(m)
			mets_pd.loc[len(mets_pd.index)] = m
			mets_pd.to_csv('mode_collapse/'+dataset+'/metrics.csv')
			all_data_returned.to_csv('ablation_study/ablation_study_data/ablation_study_all/sample_data_'+dataset+'_'+set_name+'_all.csv')
			if 'VGAN-NORM' in set_name and 'POST-FAIRGAN' not in set_name:
				all_data_returned.to_csv('ablation_study/ablation_study_data/ablation_study_all/VGAN-NORM/sample_data_'+dataset+'_'+set_name+'_all.csv')
	
	metrics_averages = mets_pd.groupby(['file']).mean().sort_values(['sim_modes'])
	metrics_averages.to_csv('mode_collapse/'+dataset+'/metrics-averages.csv')
	metrics_averages = reorganize_averages(metrics_averages)
	metrics_averages.to_csv('mode_collapse/'+dataset+'/metrics-averages.csv')
	print(metrics_averages)
	return

def get_og_modes(mets_pd, og_data):
	m = run_og(og_data)
	mets_pd.loc[len(mets_pd.index)] = m
	mets_pd.to_csv('mode_collapse/'+dataset+'/metrics.csv')
	return

if __name__ == "__main__":
	global og_kmeans_centers
	global og_kmeans

	print("EX args: [dataset] [og_file] [protected] [set_name] [set_folder] [set_name_2] [set_folder_2]")
	
	dataset = sys.argv[1]
	protected = sys.argv[2]

	## GET OLD DATA IF EXISTS
	if not os.path.exists('mode_collapse/'+dataset+'/'):
		os.makedirs('mode_collapse/'+dataset+'/')
	if os.path.isfile('mode_collapse/'+dataset+'/metrics.csv'):
		mets_pd = pd.read_csv('mode_collapse/'+dataset+'/metrics.csv', index_col=0)
	else:
		mets_pd = pd.DataFrame(columns=['file','sim_modes'])

	## TEST ORIGINAL DATA BEFORE SELECT
	og_file, og_test, VGAN_NORM, VGAN_GMM, WGAN_NORM, WGAN_GMM, VGAN_ORDI_NORM, VGAN_ORDI_GMM, TABFAIRGAN, FAIRGAN =  get_og_file(dataset, protected)
	FAIR_SWAP = "FAIR_SWAP" 
	before_select = [VGAN_NORM, VGAN_GMM, WGAN_NORM, WGAN_GMM, VGAN_ORDI_NORM, VGAN_ORDI_GMM, TABFAIRGAN, FAIRGAN, FAIR_SWAP]
	before_select_set_names = ['VGAN-norm', 'VGAN-gmm', 'WGAN-norm', 'WGAN-gmm', 'VGAN-ordi-norm', 'VGAN-ordi-gmm', 'tabfairgan', 'fairgan', 'fairswap']

	## GET OG DATA MODES
	protected, privileged, _, predicted, preferred, _ = get_labels(og_file, protected)
	og_data,_,ohe_og = get_data(og_file, og_file, protected, privileged, predicted, preferred)
	og_data_returned = return_data( og_data, ohe_og, og_file, protected, predicted, privileged, preferred )
	get_og_modes(mets_pd, og_data )

	if len(sys.argv) > 3:
		before_select = [ sys.argv[3] ] 
		before_select_set_names = [ sys.argv[4] ]

	def get_data_files(testing_folder, protected, dataset_name):
		data_files = [ testing_folder+'/'+file for file in os.listdir(testing_folder) if file.startswith('sample_data') ] 
		## CLEAN DATA FILES
		if protected == 'gender':
			data_files = [x for x in data_files if not 'race' in x]
		if protected == 'race':
			data_files = [x for x in data_files if not 'gender' in x]
			data_files = [x for x in data_files if not 'sex' in x]
		data_files = [x for x in data_files if not 'nofair' in x]
		return sorted(list(set([x for x in data_files if dataset_name in x ])))

	poss_datasets = ['adult', 'bank', 'compas', 'medical']
	dataset_name = dataset
	for pos in poss_datasets:
		if pos in dataset:
			dataset_name = pos 
	## TEST EACH DATASET BEFORE SELECT
	for i in range(len(before_select)):
		testing_folder = before_select[i] ## Get files in directory
		set_name = before_select_set_names[i].upper()
		data_files = get_data_files(testing_folder, protected, dataset_name)
		## RUN TEST
		print(set_name)
		get_modes(mets_pd, dataset, protected, set_name, data_files)
	## TEST EACH DATASET AFTER SELECT
	if 'FAIR_SWAP' in before_select:
		before_select.remove('FAIR_SWAP')
	for i in range(len(before_select)):
		set_name = before_select_set_names[i]
		if dataset_name == 'compas':
			testing_folder = 'ablation_study/ablation_study_data/ablation_study_GAN_type/'+dataset_name+'-'+protected+'-'+set_name+'_selected_samples' ## Get files in directory
		else:
			testing_folder = 'ablation_study/ablation_study_data/ablation_study_GAN_type/'+dataset_name+'-'+set_name+'_selected_samples' ## Get files in directory
		data_files = get_data_files(testing_folder, protected, dataset_name)
		print(data_files)
		set_name = 'POST-FAIRGAN-'+set_name.upper()
		print(set_name)
		## RUN TEST
		get_modes(mets_pd, dataset, protected, set_name, data_files)

	if not os.path.exists('mode_collapse/'+dataset+'/'):
		os.makedirs('mode_collapse/'+dataset+'/')

