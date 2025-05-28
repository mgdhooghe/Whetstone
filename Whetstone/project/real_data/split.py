import pandas as pd
import numpy as np
import sys
from main_transform import get_labels
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

	####################
	# Train, Valid, Test: 60,20,20 split
	####################

	file = sys.argv[1]
	prot = sys.argv[2]
	_,_,_,predicted,_,_=get_labels(file, prot)
	data = pd.read_csv(file, header='infer')#, sep=';')
	print(data)
	y = data[predicted]
	x = data.drop(columns=[predicted])
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y)#Split into Train and Test
	x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size=0.25,stratify=y_train)#Split Train into Train and Validate

	test=pd.concat([x_test,y_test],axis=1).reset_index(drop=True)
	test_file = file.replace('.csv','-SPLIT-TEST-20.csv')
	test.to_csv(test_file,index=False)

	train=pd.concat([x_train,y_train],axis=1).reset_index(drop=True)
	train_file = file.replace('.csv','-SPLIT-TRAIN-60.csv')
	train.to_csv(train_file,index=False)

	valid=pd.concat([x_valid,y_valid],axis=1).reset_index(drop=True)
	valid_file = file.replace('.csv','-SPLIT-VALID-20.csv')
	valid.to_csv(valid_file,index=False)
