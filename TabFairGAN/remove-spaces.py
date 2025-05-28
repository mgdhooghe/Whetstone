import pandas as pd
filename = 'ADULT-SPLIT-TRAIN-60.csv'
data = pd.read_csv(filename)

for col in data.columns:
	if data[col].dtypes in ['O','str']:
		print(col)
		data[col] = [x.strip() for x in data[col]]
		print(data[col].unique())

data.to_csv(filename,index=False)
