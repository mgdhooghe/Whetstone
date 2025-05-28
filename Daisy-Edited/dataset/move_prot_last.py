import pandas as pd
import sys

filename = sys.argv[1]
prot = sys.argv[2]

data = pd.read_csv(filename)
data_prot = data[prot]
data = data.drop(columns=[prot])
data = pd.concat([data, data_prot],axis=1)
data.to_csv(filename, index=False)

