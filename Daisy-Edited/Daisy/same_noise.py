import torch
from synthesizer import VGAN_generator, VGAN_discriminator
from load import get_data
from data import Dataset

##FROM RUN
args = 'C:/Users/grace_3heojyk/Desktop/Fairness_Research/Daisy-Git-Pull/Daisy/params/param-adult-short.json'
sampleloader, dataset = get_data(args)
print(sampleloader)
print(dataset)

