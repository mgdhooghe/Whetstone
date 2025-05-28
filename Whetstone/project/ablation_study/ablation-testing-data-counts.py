import sys
import os
import warnings
import pandas as pd
from main_transform import * 
from var_utils import *

def fill_x(og_data, data):
    diff = list(set(og_data.columns)-set(data.columns))
    add_cols = pd.DataFrame(0, index=np.arange(len(data)), columns=diff)
    data = pd.concat([data,add_cols],axis=1)
    return data[og_data.columns]
    
#################################################### MAIN ######################################################
if __name__ == "__main__":
    ############################
    # Get Data + One Hot Encode
    ############################
    pd.set_option('display.max_columns',None)
    directory = sys.argv[1]
    print('Synthetic Data Directory: ',directory)
    data_name = sys.argv[2]
    print('Dataset Name: ',data_name)
    training_file = sys.argv[3]
    print('Training Dataset File: ',training_file)
    protected = sys.argv[4]
    print('Protected Feature: ',protected)
    protected, privileged, unprivileged, predicted, preferred, unpreferred = get_labels(training_file, protected)
    
    # Get all Files
    og_file = training_file
    og_data,_,ohe = get_data(og_file, og_file, protected, privileged, predicted, preferred)
    og_counts = og_data.value_counts([protected,predicted], normalize=True)
    og_counts = og_counts.reset_index()
    og_counts['prot_pred'] = og_counts[protected].astype(str) + og_counts[predicted].astype(str)
    og_counts = og_counts.set_index(['prot_pred']).drop(columns=[protected,predicted]).sort_index().T.rename(index={0: 'ORIGINAL'})
    print("OG COUNTS")
    print(og_counts)
    print(len(og_data.index))
    print('25: ',len(og_data.index)*.25)
    print('75: ',len(og_data.index)*.75)

    ######## START WITH RANDOM DATA #######
    files = [ file for file in os.listdir(directory) if file.startswith('sample_data_') ]
    i = 0
    collapsed = []
    for file in files:
        print(file)
        data,_,_ = get_data(directory+"/"+file, og_file, protected, privileged, predicted, preferred)
        data = fill_x(og_data, data) 
        if len(data.drop_duplicates().index) > 1: 
            
            ## Combine all data into one dataset
            if i == 0:
                all_data = data
            else:
                all_data = pd.concat([all_data, data], axis=0, ignore_index=True) 
            i = i + 1
            all_data = all_data.astype(float).dropna(axis='index').reset_index(drop=True)
            N = int(len(all_data.index)/i) 
            val_counts = all_data.value_counts([protected,predicted])
            print("ALL COUNTS: ",val_counts)
            #print("MIN COUNTS: ",min(val_counts))
            #print("Min Percent: ",min(val_counts)/N)
            norm_val_counts = all_data.value_counts([protected,predicted], normalize=True)
            print("NORM COUNTS: ",norm_val_counts)
        else:
            collapsed = collapsed + [file] 
    #print(str(i)+'/'+str(len(files)))
    #print(sorted(collapsed))
    #for file in collapsed:
    #    os.rename(os.path.join(directory, file), os.path.join(directory, "collapsed_"+file))
    f_name = data_name+'_var_counts.txt'
    norm_val_counts = norm_val_counts.reset_index()
    norm_val_counts['prot_pred'] = norm_val_counts[protected].astype(str) + norm_val_counts[predicted].astype(str)
    norm_val_counts = norm_val_counts.set_index(['prot_pred']).drop(columns=[protected,predicted]).sort_index().T.rename(index={0: directory})
    if not os.path.isfile(f_name):
        f = open(f_name, "w")
        #f.write("ORIGINAL: ")
        f.write(str(og_counts))
        f.close()

    f = open(f_name, "a")
    #f.write(directory)
    f.write("\n"+str(norm_val_counts))
    f.close()
         

 
