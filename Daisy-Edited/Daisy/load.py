from data import NumericalField, CategoricalField, Iterator
from data import Dataset
from synthesizer import VGAN_generator, VGAN_discriminator
from synthesizer import LGAN_generator, LGAN_discriminator, LSTM_discriminator
from synthesizer import DCGAN_generator, DCGAN_discriminator
from synthesizer import V_Train,C_Train,W_Train, C_Train_dp, C_Train_nofair
from random import choice
import multiprocessing
import pandas as pd
import numpy as np
import torch
import argparse
import json
import os
import re

VGAN_variable = {
	"batch_size":[128,512,256],
	"z_dim":[128,256],
	"gen_hidden_dim":[100,200,300,400],
	"gen_num_layers":[1,2,3],
	"dis_hidden_dim":[100,200,300,400],
	"dis_num_layers":[1,2,3],
	"dis_lstm_dim":[100,200,300,400],
	"lr":[0.0001,0.0002,0.001,0.0005],
	"noise":[0.05,0.1,0.2,0.3]
}

LGAN_variable = {
	"batch_size":[50,100,200],
	"z_dim":[50,100,200,400],
	"gen_feature_dim":[100, 200, 300, 400, 500,600],
	"gen_lstm_dim":[100, 200,300,400,500,600],
	"dis_hidden_dim":[100,200,300,400,500],
	"dis_num_layers":[1,2,3,4,5],
	"dis_lstm_dim":[100,200,300,400],
	"lr":[0.0002,0.0001,0.0005,0.001],
	"noise":[0.05,0.1,0.2,0.3]
}

DCGAN_variable = {
	"batch_size":[50,100,150],
	"z_dim":[50, 100, 200, 300, 400],
	"lr":[0.0005, 0.0001, 0.0002, 0.0003]
}

def parameter_search(Model):
	parameters = {}
	if Model == "VGAN":
		variable = VGAN_variable
	elif Model == "LGAN":
		variable = LGAN_variable
	elif Model == "DCGAN":
		variable = DCGAN_variable
	for param in variable.keys():
		parameters[param] = choice(variable[param])
	return parameters

def thread_run(path, search, config, col_type, dataset, sampleset):
	#torch.multiprocessing.set_start_method('spawn')
	if config["rand_search"] == "yes":
		param = parameter_search(config["model"])
	else:
		param = config["param"]
		
	with open(path+"exp_params.json", "a") as f:
		json.dump(param, f)
		f.write("\n")
		
	model = config["model"]
	train_method = config["train_method"]

	if train_method == "CTrain" or train_method == "CTrain_dp" or train_method == "CTrain_nofair":
		print(config["label"])
		labels = config["label"]
	else:
		labels = None
	if model == "DCGAN":
		square = True
		pad = 0
	else:
		square = False
		pad = None

	#print(dataset.col_ind)
	#print(sampleset.col_ind)
	#print(labels)
	train_it, sample_it = Iterator.split(
		batch_size = param["batch_size"],
		train = dataset,
		validation = sampleset,
		sort_key = None,
		shuffle = True,
		labels = labels,
		square = square,
		pad = pad
	)
	x_dim = train_it.data.shape[1]
	col_dim = dataset.col_dim
	col_ind = dataset.col_ind
	if train_method == "CTrain" or train_method == "CTrain_dp" or train_method == "CTrain_nofair":
		c_dim = train_it.label.shape[1]
		condition = True
	else:
		c_dim = 0
		condition = False
		
	if model == "VGAN":
		gen = VGAN_generator(param["z_dim"], param["gen_hidden_dim"], x_dim, param["gen_num_layers"], col_type, col_ind, condition=condition,c_dim=c_dim)
		if "dis_model" in config.keys():
			if config["dis_model"] == "lstm":
				dis = LSTM_discriminator(x_dim, param["dis_lstm_dim"], condition, c_dim)
			elif config["dis_model"] == "mlp":
				dis = VGAN_discriminator(x_dim, param["dis_hidden_dim"], param["dis_num_layers"],condition,c_dim)
		else:
			dis = VGAN_discriminator(x_dim, param["dis_hidden_dim"], param["dis_num_layers"],condition,c_dim)
	elif model == "LGAN":
		gen = LGAN_generator(param["z_dim"], param["gen_feature_dim"], param["gen_lstm_dim"], col_dim, col_type, condition, c_dim)
		if "dis_model" in config.keys():
			if config["dis_model"] == "lstm":
				dis = LSTM_discriminator(x_dim, param["dis_lstm_dim"], condition, c_dim)
			elif config["dis_model"] == "mlp":
				dis = LGAN_discriminator(x_dim, param["dis_hidden_dim"], param["dis_num_layers"], condition, c_dim)
		else:
			dis = LGAN_discriminator(x_dim, param["dis_hidden_dim"], param["dis_num_layers"], condition, c_dim)
	#elif model == "DCGAN":
	#	gen = DCGAN_generator(param["z_dim"], train_it.shape, 2, col_type)
	#	dis = DCGAN_discriminator(train_it.shape, 2)		
		
	print(gen)
	print(dis)
	GPU = torch.cuda.is_available()

	print("GPU: ",GPU)
	if "sample_times" in config.keys():
		sample_times = config["sample_times"]
	else:
		sample_times = 1

	if train_method == "VTrain":
		print((c_dim, condition, x_dim))
		KL = True
		if "KL" in config.keys():
			KL = True if config["KL"] == "yes" else False

		####TEST G####
		z_dim = [128, 256]
		data_count = 0
		#print(sample_it.size)
		z_list_128 = None #torch.empty(sample_it.size)
		z_list_256 = None
		for x_real in sample_it:
			print('x_real: ', x_real)
			if z_list_128 is None: 
				z_list_128 = torch.randn(x_real.shape[0], 128)
			else:
				z_list_128 = torch.cat((z_list_128,torch.randn(x_real.shape[0],128)),0)
			if z_list_256 is None:
				z_list_256 = torch.randn(x_real.shape[0], 256)
			else:
				z_list_256 = torch.cat((z_list_256, torch.randn(x_real.shape[0],256)),0)
			data_count = data_count+1
			#G_paths = ["C:/Users/grace_3heojyk/Desktop/Fairness_Research/myGAN/G_0_9", "C:/Users/grace_3heojyk/Desktop/Fairness_Research/myGAN/G_1_9"]
		
		for i in range(0,7):
			sample_data = None
			k = 0
			this_G = "C:/Users/grace_3heojyk/Desktop/Fairness_Research/myGAN/G_"+str(i)+"_9"
			for x_real in sample_it:
				size = x_real.shape[0]
				num = re.findall(r'\d+', this_G)
				G = torch.load(this_G,map_location=torch.device('cpu') )
				try:
					x_fake = G(z_list_128[k*size:(k+1)*size,:])
				except:
					x_fake = G(z_list_256[k*size:(k+1)*size,:])
				x_fake = x_fake.reshape(x_fake.shape[0], x_fake.shape[1])
				k = k+1
				samples = x_fake
				samples = samples.reshape(samples.shape[0], -1)
				samples = samples[:,:train_it.dataset.dim]
				samples = samples.cpu()
				sample_table = train_it.dataset.reverse(samples.detach().numpy())
				df = pd.DataFrame(sample_table,columns=train_it.dataset.columns)
				print(path)
			
				if sample_data is None:
					sample_data = df
				else:
					sample_data = sample_data.append(df)
			
			sample_data.to_csv(path+'same_noise_data_{}_{}.csv'.format(i,num[-1]), index = None)
			print(df)
		sys.exit()
		####END TEST G####

		#V_Train(search, path, sample_it, gen, dis, config["n_epochs"], param["lr"], train_it, param["z_dim"], dataset, col_type, sample_times,itertimes = 100, steps_per_epoch = config["steps_per_epoch"],GPU=GPU,KL=KL,data=config["data"],protected=config["protected_var"])
	elif train_method == "CTrain":
		print((c_dim, condition, x_dim))
		print(train_it.label.shape)	
		C_Train(search, path, sample_it, gen, dis, config["n_epochs"], param["lr"], train_it, param["z_dim"], dataset, col_type, sample_times,itertimes = 100, steps_per_epoch = config["steps_per_epoch"],GPU=GPU,data=config["data"],protected=config["protected_var"])
	elif train_method == "WTrain":
		dis.wgan = True
		KL=True
		if "KL" in config.keys():
			KL = True if config["KL"] == "yes" else False
		W_Train(search, path, sample_it, gen, dis, config["ng"], config["nd"], 0.01, param["lr"], train_it, param["z_dim"], dataset, col_type,sample_times, itertimes=100, GPU=GPU,KL=KL)
	elif train_method == "CTrain_dp":
		dis.wgan = True
		C_Train_dp(search, path, sample_it, gen, dis, config["ng"], config["nd"], 0.01, param["lr"], train_it, param["z_dim"], dataset, col_type, config["eps"], sample_times,itertimes = 100, GPU=GPU)
	elif train_method == "CTrain_nofair":
		C_Train_nofair(search, path, sample_it, gen, dis, config["n_epochs"], param["lr"], train_it, param["z_dim"], dataset, col_type,sample_times,itertimes = 100, steps_per_epoch = config["steps_per_epoch"], GPU=GPU,data=config["data"],protected=config["protected_var"])		

if __name__ == '__main__':
	args = 'C:/Users/grace_3heojyk/Desktop/Fairness_Research/Daisy-Git-Pull/Daisy/params/param-adult-short.json'
	parser = argparse.ArgumentParser()
	parser.add_argument('configs', help='a json config file')
	f = open(args)
	configs = json.load(f)
		
	try:
		os.mkdir("expdir")
	except:
		pass
		
	for config in configs:
		path = "expdir/"+config["name"]+"/"
		try:
			os.mkdir("expdir/"+config["name"])
		except:
			pass
		train = pd.read_csv(config["train"])
		fields = []
		col_type = []

		if "ratio" in config.keys():
			ratio = config["ratio"]
		else:
			ratio = 1

		if "label" in config.keys():
			cond = config["label"]
		noise = choice([0.05,0.1,0.2,0.3])
		for i, col in enumerate(list(train)):
			if "label" in config.keys() and col in cond:
				fields.append((col, CategoricalField("one-hot", noise=0)))
				col_type.append("condition")
			elif i in config["normalize_cols"]:
				fields.append((col,NumericalField("normalize")))
				col_type.append("normalize")
			elif i in config["gmm_cols"]:
				fields.append((col, NumericalField("gmm", n=5)))
				col_type.append("gmm")
			elif i in config["one-hot_cols"]:
				fields.append((col, CategoricalField("one-hot", noise=noise)))
				col_type.append("one-hot")
			elif i in config["ordinal_cols"]:
				fields.append((col, CategoricalField("dict")))
				col_type.append("ordinal")
			else:
				fields.append((col, CategoricalField("binary",noise=noise)))
				col_type.append("binary")

		trn, samp = Dataset.split(
			fields = fields,
			path = ".",
			train = config["train"],
			validation = config["sample"],
			format = "csv",
			valid_ratio=ratio
		)
		trn.learn_convert()
		samp.learn_convert()

		print("train row : {}".format(len(trn)))
		print("sample row: {}".format(len(samp)))
		thread_run(path, 0, config, col_type, trn, samp)
		#return config, sample_it, trn
		#jobs = [multiprocessing.Process(target=thread_run, args=(path, search, config, col_type, trn, samp)) for search in range(n_search)]	
		#for j in jobs:
		#	j.start()
		#for j in jobs:
	    #	j.join()

