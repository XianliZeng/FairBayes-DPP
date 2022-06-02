import time
import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from models import Classifier
from dataloader import FairnessDataset
from algorithm import FairBayes_DPP


#####multi-class protected attribute or not##########


##### Model specifications #####
n_layers = 3 # [positive integers]
n_hidden_units = 32 # [positive integers]


##### Which dataset to test and which fairness notion to consider#####



dataset_name = 'AdultCensus'  # ['AdultCensus',  'COMPAS']
    ##### predetermine disparity level #####

##### Other training hyperparameters #####
if dataset_name == 'AdultCensus':
    n_epochs = 200
    lr = 1e-1
    batch_size = 512

if dataset_name == 'COMPAS':
    n_epochs = 500
    lr = 5e-4
    batch_size = 2048


##### Whether to enable GPU training or not
device = torch.device('cuda' if torch.cuda.is_available()==True else 'cpu' )

n_seeds = 100  # Number of random seeds to try




resultprop_PP = pd.DataFrame()
resultprop_PP_base = pd.DataFrame()
starting_time = time.time()

for seed in range(n_seeds):
    print('Currently working on - seed: {}'.format(seed))
    seed = seed * 5
    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import dataset

    dataset = FairnessDataset(dataset=dataset_name, device=device)
    dataset.normalize()
    input_dim = dataset.XZ_train.shape[1]

    # Create a classifier model
    net = Classifier(n_layers=n_layers, n_inputs=input_dim, n_hidden_units=n_hidden_units)
    net = net.to(device)

    # Set an optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Fair classifier training

    [temp_PP,temp_PP_base] = FairBayes_DPP(dataset=dataset,dataset_name=dataset_name,
                                 net=net,
                                 optimizer=optimizer,
                                 device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)

    resultprop_PP = resultprop_PP.append(temp_PP)
    resultprop_PP_base = resultprop_PP_base.append(temp_PP_base)

    print(resultprop_PP)
    print(resultprop_PP_base)

print('Average running time: {:.3f}s'.format((time.time() - starting_time) / n_seeds))