# Whetstone
Run the following line to apply Whetstone to synthetic data in the specified 'directory' with respect to the provided protected feature:  
`python main.py [directory] [training_file] [protected feature] [privileged value] [predicted feature] [preferred value] [selected_data_percentage]`

## Example
`python main.py example/german-synthetic/ example/GERMAN-SPLIT-TRAIN-60.csv gender female labels 1 .5`
