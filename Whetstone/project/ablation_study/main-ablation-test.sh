#!/bin/bash

## [directory] [dataset_name] [real$size_file] [protected feature] [privileged value] [predicted feature] [preferred value] [flag (-b/-s/-p)] [abl]

cd ..
python_code=main-ablation.py

betas=("0" "1" "25" "50" "100") #normal=10 (run in test GAN type VGAN-norm)
betas=("5")
set_percents=("100" "50" "25" "10" "1" ".5") #normal=5
percents=("0" ".20" ".5" "1") #normal=.8 (run in test GAN type VGAN-norm)
hyp_disps=("None") # "EqOp" "AvgOdd" "both" "disp_nopred") #"EqOp" "EqOp_and_Disp") #normal="disp" (run in test GAN type VGAN-norm)
#avail_nums=(".1" ".25" ".5" ".75") #normal=All
avail_nums=(".9")
try_nums=("1" "2" "15" "100") #normal=30



size=-SPLIT-VALID-20

flag=-$1
if [[ "$flag" ==  "-b" ]]; then
	echo "BETA"
	ablations=("${betas[@]}")
elif [[ "$flag" == "-sp" ]]; then 
	echo SET_PERCENT
	ablations=("${set_percents[@]}")
elif [[ "$flag" == "-p" ]]; then
	echo "PERCENT"
	ablations=("${percents[@]}")
elif [[ "$flag" == "-d" ]]; then
        echo "DISP"
        ablations=("${hyp_disps[@]}")
elif [[ "$flag" == "-a" ]]; then
        echo "MIN_ACC"
        ablations=("${min_acc[@]}")
elif [[ "$flag" == "-n" ]]; then
        echo "AVAIL_NUM"
        ablations=("${avail_nums[@]}")
elif [[ "$flag" == "-t" ]]; then
        echo "TRY_NUM"
        ablations=("${try_nums[@]}")
else 
	echo "Please select a flag value from (s/b/p/sp/d/a/n/t)"
	exit
fi

echo "Ablations: ${ablations[@]}"
#run_datasets=("german" "compas-race" "compas-gender" "medical" "bank" "adult")
#run_datasets=("compas-race" "medical" "bank" "adult")
#run_datasets=("compas-race" "medical")
run_datasets=("adult")


for ab in "${ablations[@]}";
do
	echo "Run Set: $run_datasets[@]}"
	for run_set in "${run_datasets[@]}";
	do
		echo "Flag: " $flag
		echo "Value: " $ab
		echo "Run Set: " $run_set

		if [[ "$run_set" == "german" ]]; then
        
			## German
                
			dataset=german
			real=real_data/german/GERMAN
			prot=gender
			priv=female
			pred=labels
			pref=1
			synth_data=../german/german-VGAN-1hot-split-validation-best/9/
                
			python $python_code $synth_data $dataset-VGAN-norm$flag-$ab $real$size.csv $prot $priv $pred $pref $flag $ab
        

		elif [[ "$run_set" == "compas-race" ]]; then

			## Compas-Race
                
			dataset=compas-race
			real=real_data/propublica-compas/PROPUBLICA-COMPAS
			prot=race
			priv=Caucasian
			pred=two_year_recid
			pref=0
			synth_data=../gan_data/compas/compas-pro-VGAN-1hot-norm-split-valid-best/9/
                
			python $python_code $synth_data  $dataset-VGAN-norm$flag-$ab $real$size.csv $prot $priv $pred $pref $flag $ab
        

		elif [[ "$run_set" == "compas-gender" ]]; then

			## Compas-Gender
                
			dataset=compas-gender
			real=real_data/propublica-compas/PROPUBLICA-COMPAS
			prot=sex
			priv=Female
			pred=two_year_recid
			pref=0
			synth_data=../gan_data/compas/compas-pro-VGAN-1hot-norm-split-valid-best/9/
                
			python $python_code $synth_data  $dataset-VGAN-norm$flag-$ab $real$size.csv $prot $priv $pred $pref $flag $ab
        

		elif [[ "$run_set" == "medical" ]]; then

			## Medical
                
			dataset=medical
			real=real_data/medical/meps21
			prot=RACE
			priv=1
			pred=UTILIZATION
			pref=1
			synth_data=../gan_data/medical/medical-VGAN-1hot-norm-split-60/8-9/
                
			python $python_code $synth_data $dataset-VGAN-norm$flag-$ab $real$size.csv $prot $priv $pred $pref $flag $ab
        

		elif [[ "$run_set" == "bank" ]]; then

			## Bank
                
			dataset=bank
			real=real_data/bank/bank-full
			prot=age
			priv=1
			pred=y
			pref=yes
			synth_data=../gan_data/bank/bank-VGAN-best/9-age/
                
			python $python_code $synth_data $dataset-VGAN-norm$flag-$ab $real$size-age.csv $prot $priv $pred $pref $flag $ab

		elif [[ "$run_set" == "adult" ]]; then

			## Adult
                
			dataset=adult
			real=real_data/adult/ADULT
			prot=sex
			priv=Male
			pred=income
			pref=">50K"
			synth_data=../gan_data/adult/adult-VGAN-1hot-norm-split-60/9/
			#synth_data=../gan_data/adult/adult-VGAN-1hot-gmm-split-60-WTrain-best/90/

			python $python_code $synth_data  $dataset-VGAN-norm$flag-$ab $real$size.csv $prot $priv $pred $pref $flag $ab
			#python $python_code $synth_data  $dataset-WGAN-gmm$flag-$ab $real$size.csv $prot $priv $pred $pref $flag $ab
		fi
	done

done
