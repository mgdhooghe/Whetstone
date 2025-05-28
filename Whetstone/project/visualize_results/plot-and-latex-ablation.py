import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib
from random import sample
from matplotlib import pyplot as plt

import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 15}
font = {'weight' : 'bold',
        'size'   : 15}
matplotlib.rc('font', **font)


### ORGANIZE DATA ###
def organize_data(directory, file, ablation):
	print(directory+'/'+file)
	data = pd.read_csv(directory+'/'+file, index_col=0) #-1)
	data.fillna(0, inplace=True)

	####################
	#Organize data 
	####################
	dataset_name = ''
	protected = ''
	og_name = ''
	if 'adult' in directory:
	    dataset_name = 'Adult'
	    og_name = 'ADULT-SPLIT-TRAIN-'
	    protected = 'gender'
	elif 'compas' in directory:
	    dataset_name = 'Compas'
	    og_name = 'PROPUBLICA-COMPAS-SPLIT-TRAIN-'
	    if 'gender' in directory:
	        protected = 'gender'
	    else:
	        protected = 'race'
	    protected = 'race'
	elif 'german' in directory:
	    dataset_name = 'German'
	    og_name = 'GERMAN-SPLIT-TRAIN-'
	    protected = 'gender'
	elif 'bank' in directory:
	    dataset_name = 'Bank'
	    og_name = '59'
	    protected = 'age'
	elif 'medical' in directory:
	    dataset_name = 'Medical'
	    og_name = '20'
	    protected = 'race'

	## GAN DATA
	if dataset_name != 'Compas': select_folder = "mode_collapse/"+dataset_name.lower()+"_SELECT_FROM/selection-metrics-averages.csv"
	else: select_folder = "mode_collapse/compas-race_SELECT_FROM/selection-metrics-averages.csv"
		
	select_from = pd.read_csv(select_folder, index_col=0)
	if dataset_name != 'Adult' or "VGAN-norm" in directory: select_from = select_from["VGAN-norm"]	
	else: select_from = select_from["WGAN-gmm"]	
	
	select_from.name = "GAN"

	## RENAME BASED ON ABLATION
	data.rename(index={og_name:'OG'},inplace=True)
	data.index = [ 'GAN' if re.sub('_\d+_\d+_\d+.csv','',txt) == 'sample_data' else txt for txt in data.index ]
	data.index = [str(txt).lower().replace('-','_').replace('sample_data_parallel_','').replace('sample_data_','').replace('meps_','').replace('_split','').replace(dataset_name.lower(),'').replace('_'+protected+'_','').replace('_'+protected,'').replace('_age','').replace('race','').replace('gender','').replace('sex','').replace('pro','').replace('_','',1).replace('_','',-1) for txt in data.index]
	if 'gan_type' not in ablation.lower():
		data.index = [str(txt).lower().replace('vgannorm','').replace('_','',1) for txt in data.index]
		#data.index = [ 'base' if txt == 'vgan_norm' else txt for txt in data.index ]
		#data.index = [ 'base' if txt == '' else txt for txt in data.index ]
	if 'FAIR_SWAP' in ablation or 'ALTERNATIVES' in ablation:
		data.index = [ 'POST-FairGAN' if txt == 'base' else txt for txt in data.index ]
		data.index = [ 'TabFairGAN' if txt == 'tab' else txt for txt in data.index ]
		data.index = [ 'FairGAN' if txt == 'fairgan' else txt for txt in data.index ]
		data.index = [ 'FAIR_SWAP_BEFORE_EVEN' if 'nofair' in txt else txt for txt in data.index ]
		if 'FAIR_SWAP_BEFORE_EVEN' in data.index.array:	
			data = data.drop(index='FAIR_SWAP_BEFORE_EVEN')
		data.index = [ 'Fair_Swap' if txt == 'vgannorm' else txt for txt in data.index ]
		data.index = [ 'Fair_Swap' if 'fairswap' in txt else txt for txt in data.index ]
		data.index = [ 'NAIVE1' if 'all' in txt else txt for txt in data.index ]
		if 'NAIVE1' in data.index.array:
			data = data.drop(index=['NAIVE1'])
		data.index = [ 'POST-FairGAN NAIVE' if '2' in txt else txt for txt in data.index ]
	#data.index = [ 'base' if txt == '' else txt for txt in data.index ]
	data.index = [ '1' if txt == '' else txt for txt in data.index ]
	if 'select_percent' in ablation: 
		data.index = [ '.8' if txt == 'base' else txt for txt in data.index ]
		data.index = [ str(int(100*float(txt))) if txt[0].isdigit() or txt[-1].isdigit() else txt for txt in data.index ]
		for val in ['70','75','90']:
			if val in data.index.array:
				data = data.drop(index=[val])
	elif 'set_percent' in ablation: 
		data.index = [ '5' if txt == 'base' else txt for txt in data.index ]
		data = data.drop(index=('.5'))
	elif 'min_acc_bound' in ablation: 
		data.index = [ '0' if txt == 'base' else txt for txt in data.index ]
	elif 'beta' in ablation: 
		data.index = [ '10' if txt == 'base' else txt for txt in data.index ]
		if dataset_name == 'Adult':
			keep = [ ('wgangmm' in x) | (x == '10') | ('original' in x) for x in data.index ]
			data = data.loc[keep]
			data.index = [ x.replace('wgangmmb','') for x in data.index ]
		else:
			data.index = [ x.replace('b','') for x in data.index ]
		print(data.index)
	elif 'avail_num' in ablation:
		data = data[data.index != 'base']
		data = data[data.index != 'Base']
		data.index = [ txt.replace('n.','.') for txt in data.index ]
		print(data)
		data.index = [ str(float(txt)*100) if txt[0].isdigit() or txt[-1].isdigit() else txt for txt in data.index ]
	elif 'try_num' in ablation:
		data.index = [ x.replace('wgangmmt','') for x in data.index ]
		data.index = [ x.replace('t','') for x in data.index ]
		data.index = [ '30' if txt == 'base' else txt for txt in data.index ]
	elif 'no_test_split' in ablation:
		data.index = [ '1' if txt == 'base' else txt for txt in data.index ]
	elif 'hyp_disp' in ablation:
		data.index = [ 'Disp' if txt == 'base' else txt for txt in data.index ]
		data.index = [ 'None' if txt == 'dnone' else txt for txt in data.index ]
		data.index = [ 'EqOpp' if txt == 'deqop' else txt for txt in data.index ]
		data.index = [ 'AvgOdds' if txt == 'davgodd' else txt for txt in data.index ]
	else:
		data.index = [ '0' if txt == 'base' else txt for txt in data.index ]

	if 'average' in file:
		data = data.groupby(level=-1).mean()

	return data, dataset_name, og_name, select_from, protected 

### PLOT DATA ###
def plot_data(data_folders, data_files, acc_measure, ablation, dir):
	start = True 
	leg = False #True
	for directory in data_folders:
		data_files = ["metrics-averages.csv"]
		fig, ax = plt.subplots(len(data_files) ,1, sharex=True, sharey=True)
		if len(data_files) > 1: XLAB = ax[len(data_files)-1].set_xlabel('Disparate Impact', fontweight="bold")
		else: XLAB = ax.set_xlabel('Disparate Impact', fontweight="bold")
			
		i = 0
		for file in data_files:
			if not os.path.isfile(directory+'/'+file): break
				
			if len(data_files) > 1: ax_alias = ax[i]
			else: ax_alias = ax
				
			if acc_measure == 'bal_acc': ax_alias.set_ylabel('Balaced Accuracy', fontweight="bold")
			else: ax_alias.set_ylabel('Accuracy', fontweight="bold")
				
			data, dataset_name, og_name, gan_data, protected = organize_data(directory, file, ablation)
			if dataset_name == 'Adult' and 'VGAN-norm' in directory: TITLE = ax_alias.set_title(dataset_name+': VGAN Norm', fontweight="bold")
			elif dataset_name == 'Adult': TITLE = ax_alias.set_title(dataset_name+': WGAN GMM', fontweight="bold")
			else: TITLE = ax_alias.set_title(dataset_name, fontweight="bold")
				

			####### PLOT SAVED AS ########
			fig_name = dir+'/'+ablation+'_'+dataset_name+'_'+protected+'_'+acc_measure+'_plot.eps'
			lgnd_name = dir+'/'+ablation+'_'+dataset_name+'_'+protected+'_'+acc_measure+'_lgnd.eps'
			
			#########################
			# Plot Average Metrics
			#########################
			
			data['color'] = 'C0'
			c_i = 1
			for uind in sorted(data.index.unique()):
				data.loc[uind,'color'] = 'C'+str(c_i)
				c_i += 1
			
			## PLOT ERROR BARS
			og_data = data[data.index == 'original']
			if 'gan' in data.index.array: data = data.rename(index={'gan':'GAN'})
			if 'GAN' in data.index.array: gan_data = data[data.index == 'GAN'].iloc[0]
			data = data[data.index != 'original']
			data = data[data.index != 'GAN']

			if not 'GAN_TYPE' in ablation and not 'ALTERNATIVES' in ablation and not 'hyp_disp' in ablation:
				data.index = [ int(eval(i)) for i in data.index ]
				data = data.sort_index(axis=0)
                
			if 'beta' in ablation:
				keep = [ ind < 1000 or ind >= float(10**10) for ind in data.index ]
				data = data[keep]

			if "GAN" not in data.index.array:
				data = pd.concat([gan_data.to_frame().T,data])	

			data.index = [str(txt) for txt in data.index]
			if 'ALTERNATIVE' in ablation:
				data_order = ['GAN','FairGAN','TabFairGAN','Fair_Swap','POST-FairGAN NAIVE','POST-FairGAN']
				print(data_order)
			else:
				data_order = data.index.array
			c_i = 1
			marker_size = 90 
			markers_lst = ['o','s','^','v','D','p','h']
			ax_alias.scatter(og_data['DI'].iloc[0], og_data[acc_measure].iloc[0], color='k', s=marker_size)
			for uind in data_order:
				if 'DI_sem' in data.columns:
					scatter_plot = ax_alias.errorbar(data.loc[uind,'DI'],data.loc[uind,acc_measure], xerr=data.loc[uind,'DI_sem'], yerr=data.loc[uind, acc_measure+'_sem'], fmt=markers_lst[c_i-1], ecolor='C'+str(c_i),c='none')
				if np.count_nonzero(data.index.array==uind) > 5:
					plot_sample = data.loc[uind].sample(5)
				else:
					plot_sample = data.loc[uind]
				scatter_plot = ax_alias.scatter(plot_sample['DI'],plot_sample[acc_measure],c='C'+str(c_i), s=marker_size)
				c_i += 1
			box = ax_alias.get_position()
			ax_alias.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
			if leg:
				try:
					lgd = plt.legend(['Original']+list(data_order), loc='lower center', ncol=8, bbox_to_anchor=(0.5,-0.35), columnspacing=.6, handletextpad=0.1)
                                
				except Exception as e:
					print(e)
			ax_alias.axhline(y=og_data[acc_measure].iloc[0],color='k', linestyle='dashed')
			ax_alias.axvline(x=og_data['DI'].iloc[0], color='k', linestyle='dashed')
			ax_alias.annotate('Og', (og_data['DI'].iloc[0], og_data[acc_measure].iloc[0]))
			## ANNOTATE POINTS
			'''
			if 'average' in file:
				for k, txt in enumerate(data.index):	
					txt = str(txt).lower().replace('sample_data_parallel_','').replace(dataset_name.lower(),'').replace('-'+protected,'').replace('_'+protected+'_','').replace('_sex_','').replace('-vgan-norm','')
					if txt == '':
						txt = 'Base'
					ax_alias.annotate(str(txt).lower(), (data['DI'].iloc[k], data[acc_measure].iloc[k]))
			'''
			## NEXT SUBPLOT
			i = i + 1

			## SAVE DATA
			if 'original' not in [ x for x in data.index ]:
				data = pd.concat([data,og_data])
			data['dataset'] = dataset_name
			if start and file == "metrics-averages.csv":
				collected_data = data
				start = False
			elif file == "metrics-averages.csv":
				collected_data = pd.concat([collected_data, data])
			## SAVE PLOT
			if leg:
				fig.savefig(fig_name, bbox_extra_artists=(lgd,TITLE,XLAB,),bbox_inches='tight', dpi=1200)
				fig_legend = plt.figure(figsize=(3,2))
				ax_legend = legend.add_subplot(111)
				ax_legend.legend(*legend.legendHandles, loc='center')
				ax_legend.axis('off')	
				fig_legend.savefig(lgnd_name,bbox_inches='tight',dpi=1200)
			else:
				fig.savefig(fig_name, bbox_extra_artists=(TITLE,XLAB,),bbox_inches='tight')
			## CLEAR PLOT
			plt.close()
			plt.clf()

	return collected_data

########################### LATEX ############################
def paper_latx(latx_data, latx_label, f, write, last=False):
	## BOLD BEST VALUES
	def max_bold(data, props=''):
		best = 'min'
		int_data = data.copy(deep=True)
		int_data.loc['Dataset'] = str(1)
		if 'acc' in latx_label or 'rec' in latx_label or 'F1' in latx_label:
			## BOLD ONLY MAX IGNORE ORIGINAL
			int_data['Original']=str(0)
			best = 'max'
			int_data.loc['Dataset'] = str(0)
			int_data = int_data.replace('nan',str(0))
		else:
			int_data = int_data.replace('nan',str(1))
			int_data['Original']=str(1)
		int_data = int_data.map(lambda x: x.replace('-',''))
		## FIND BEST
		if best == 'max': is_best = np.max(int_data) == data
		else: is_best = np.min(int_data) == data

		return np.where(is_best, 'bfseries:', '')

	## WRITE TO FILE 
	latx_data = latx_data.reset_index()
	latx_data.columns = [ x.replace('_','-') for x in latx_data.columns ]
	print("TO FILE: ",f)
	with open(f,write) as tf:
		if write == 'w':
			if len(latx_data.columns) > 4: tf.write('\\begin{table*}\\resizebox{\\textwidth}{!}{\n')
			else: tf.write('\\begin{table}\n')
				
			cap = f.split('/')[-1].split('.')[0]
			tf.write('\caption{'+cap+'\label{table: '+latx_label+'}}\n')
			tf.write('\\centering\n')
			#tf.write('\\tiny\n')
		s=latx_data.astype(str).style.hide()
		s=s.apply(max_bold, axis=1)
		#tf.write(s.to_latex(multirow_align="t",hrules=True))
		print(latx_data)
		tf.write(s.to_latex(hrules=True))
		if last: 
			if len(latx_data.columns) > 4: tf.write('}\\end{table*}')
			else: tf.write('}\\end{table}')
				
	return

#######MAIN#################
ablation = sys.argv[1]

##########################
#Read in data from select
##########################
directory = 'mode_collapse'
#directory = 'mode_collapse/adult_avail_num'
data_files = ["metrics-averages.csv", "metrics-plot-all.csv" ]
#acc_measures = ["bal_acc", "acc"]
acc_measures = ["bal_acc"]

## GET ALL DATA
data_folders = [ directory+'/'+folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory,folder)) ]
data_folders = [folder for folder in data_folders if folder.endswith(ablation)]
data_folders.sort()

data_dict = {}
##############
## PLOT DATA
##############
for acc_measure in acc_measures: data_dict[acc_measure] = plot_data(data_folders, data_files, acc_measure, ablation, directory)

## ROUND ALL VALUES
def round_float(value):
	if isinstance(value, float): return round(value,3)
	else: return value

def calculate_f1(data):
    rec = data['rec']
    prec = data['prec']
    f1 = 2*(prec*rec)/(prec+rec)
    return f1


## CALCULATE F1
data_dict['bal_acc']['F1'] = data_dict['bal_acc'][['dataset','rec','prec']].apply(calculate_f1, axis=1)
data_dict['bal_acc']['F1_sem'] = 0
## ROUND DATA
data_dict['bal_acc'] = data_dict['bal_acc'].map(round_float)
for metric in ['DI', 'bal_acc', 'acc', 'Stat Par', 'Eq Opp', 'Avg Odds', 'Theil', 'rec', 'prec','F1']:
	## GET AVERAGE DATA
	di_data = data_dict['bal_acc'][['dataset',metric,metric+'_sem']]
	di_data = di_data[di_data['dataset'] != 'German']
	## COMBINE STANDARD ERROR DATA AND AVERAGE DATA
	di_data = di_data.map(str)
	di_data_og = di_data.loc['original'] 
	if metric != 'F1': di_data[metric] = di_data[metric].str.cat(di_data[metric+'_sem'], sep=" $\pm$ ")
	di_data.loc['original'] = di_data_og
	di_data = di_data.drop(columns=[metric+'_sem'])
	di_data = di_data.reset_index().pivot(index='dataset', columns='index',values=metric)
	di_data.columns = [ '\sys' if col == "POST-FairGAN" else col for col in di_data.columns ]
	di_data.columns = [ 'naive' if 'naive' in col.lower() else col for col in di_data.columns ]
	if 'ALTERNATIVES' in ablation:
		di_data.columns = [ x.lower() for x in di_data.columns ]
		if metric in ['DI','bal_acc','acc','rec','prec','F1']:
			cols = [ 'original', 'gan', 'fairgan', 'tabfairgan', 'fair_swap', 'naive', '\sys' ]
			cols = [ x for x in cols if x in di_data.columns ]
			di_data = di_data[cols]
		else: di_data = di_data[[ 'gan', '\sys' ]] # 'original'
		di_data.columns = [ str(x).title() if '\sys' not in x else str(x) for x in di_data.columns]
	else:
		di_data.columns = [ float(str(x)) if str(x)[0].isdigit() or str(x)[0] == '.' else x for x in di_data.columns ]
		not_og = sorted([ x for x in di_data.columns if x != 'original' and x != 'GAN' ])
		if ablation != 'beta': di_data = di_data[ [ 'original' ] + not_og ]
		else: di_data = di_data[ [ 'original', 'gan' ] + not_og ]
		di_data.columns = [ str(x).title() if type(x) == type('a') else str(int(x)) for x in di_data.columns]
	di_data.index.names = ['Dataset']

	##################
	## GET LATX FORMAT
	##################
	table_name = metric
	if ablation != 'ALTERNATIVES':
		f = directory+'/'+ablation+'_'+metric+'_latx.tex'
		table_name = ablation+"-"+metric
	elif metric == 'acc': f = directory+'/Accuracy.tex'
	elif metric == 'bal_acc': f = directory+'/AccuracyBal.tex'
	elif metric == 'DI': f = directory+'/DisparateImpact.tex'
	elif metric == 'Avg Odds': f = directory+'/AvgOdds.tex'
	elif metric == 'Stat Par': f = directory+'/StatisticalPar.tex'
	elif metric == 'Eq Opp': f = directory+'/EqualOpp.tex'
	elif metric == 'Theil': f = directory+'/Theil.tex'
	elif metric == 'rec': f = directory+'/Recall.tex'
	elif metric == 'prec': f = directory+'/Precision.tex'
	elif metric == 'F1': f = directory+'/F1.tex'
		
	paper_latx(di_data, table_name,f,'w',True)
