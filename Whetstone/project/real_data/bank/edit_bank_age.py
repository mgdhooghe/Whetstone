import pandas as pd
import numpy as np
import sys

file = sys.argv[1]
data = pd.read_csv(file)#, delimiter=';')
print(data)
file = str.split(file,'.csv')[0]+'-age.csv'
data['age'] = np.where((data['age'] >= 25) & (data['age'] < 60),1,0)
print(sum(data['age']))
print(len(data.index))
print(file)
print(data)
data.to_csv(file, index=False)
