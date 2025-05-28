#!/bin/bash

#datasets=('adult' 'bank' 'compas-race' 'compas-gender' 'german' 'medical')
#protected=('gender' 'age' 'race' 'gender' 'gender' 'race')
#ablations=('hyp_disp' 'beta' 'set_percent' 'select_percent' 'GAN_type')

#datasets=('adult' 'bank' 'compas-race' 'compas-race' 'medical')
#protected=('gender' 'age' 'race' 'gender' 'race')
#protected_selected=('gender' 'age' 'race' 'gender' 'race')

datasets=('compas-race' 'adult')
protected_selected=('race' 'gender') #Value data was selected on
protected=('gender' 'race') #Value to Test on 

ablations=('beta' 'select_percent' 'set_percent' 'avail_num' 'try_num' 'hyp_disp')
#for a in ${ablations[@]}; do
	#for i in ${!datasets[@]}; do
		#python testing-metrics.py ${datasets[$i]}_$a ${protected[$i]} ${protected_selected[$i]} ablation_study/ablation_study_data/ablation_study_$a/
	#done
	#python plot-and-latex-ablation.py $a
#done

## GAN TYPE
#for i in ${!datasets[@]}; do
#	python testing-metrics.py ${datasets[$i]}_GAN_type ${protected[$i]} ablation_study/ablation_study_data/ablation_study_GAN_type/
#done
#python plot-select-from-and-latex-ablation.py GAN_type

## ALTERNATIVES
for i in ${!datasets[@]}; do
	# NAIVE 1 
	#echo "NAIVE 1"
	#python testing-metrics.py ${datasets[$i]}_ALTERNATIVES ${protected[$i]} ablation_study/ablation_study_data/ablation_study_all/VGAN-NORM/
	# NAIVE 2
	echo "NAIVE 2"
	python testing-metrics.py ${datasets[$i]}_ALTERNATIVES ${protected[$i]} ${protected_selected[$i]} ablation_study/ablation_study_data/ablation_study_NAIVE2/
	#echo "FAIR SWAP"
	#python testing-metrics.py ${datasets[$i]}_ALTERNATIVES ${protected[$i]} ${protected_selected[$i]} FAIR_SWAP/
	#echo "FAIRGAN"
	#python testing-metrics.py ${datasets[$i]}_ALTERNATIVES ${protected[$i]} ${protected_selected[$i]} FAIRGAN/
	#echo "TABFAIRGAN"
	#python testing-metrics.py ${datasets[$i]}_ALTERNATIVES ${protected[$i]} ${protected_selected[$i]} TABFAIRGAN/
done
#python plot-and-latex-ablation.py ALTERNATIVES
