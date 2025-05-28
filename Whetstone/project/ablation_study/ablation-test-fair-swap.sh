#!/bin/bash


python_code=fair-swap.py
size=-SPLIT-TRAIN-60

## German
dataset=german
real=german/GERMAN
prot=gender
priv=female
pred=labels
pref=1

#python $python_code ../german/german-VGAN-1hot-split-validation-best/9/ $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 

## Compas-Race

dataset=compas-race
real=propublica-compas/PROPUBLICA-COMPAS
prot=race
priv=Caucasian
pred=two_year_recid
pref=0

#python $python_code ../gan_data/compas/compas-pro-VGAN-1hot-norm-split-valid-best/9/  $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 
## Compas-Gender

dataset=compas-gender
prot=sex
priv=Female

#python $python_code ../gan_data/compas/compas-pro-VGAN-1hot-norm-split-valid-best/9/  $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 

## Medical

dataset=medical
real=medical/meps21
prot=RACE
priv=1
pred=UTILIZATION
pref=1

#python $python_code ../gan_data/medical/medical-VGAN-1hot-norm-split-60/8-9/  $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 

## Bank

dataset=bank
real=bank/bank-full
prot=age
priv=1
pred=y
pref=yes

#python $python_code ../gan_data/bank/bank-VGAN-best/9-age/  $dataset-VGAN-norm $real$size-age.csv $prot $priv $pred $pref 

## Adult

dataset=adult
real=adult/ADULT
prot=sex
priv=Male
pred=income
pref='>50K'

#python $python_code ../gan_data/adult/adult-VGAN-1hot-norm-split-60/9/  $dataset-VGAN-norm $real$size.csv $prot $priv $pred $pref 
python $python_code ../gan_data/adult/adult-VGAN-1hot-gmm-split-60-WTrain-best/90/  $dataset-VGAN-gmm $real$size.csv $prot $priv $pred $pref 


