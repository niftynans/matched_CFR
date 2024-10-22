import hydra
import os
from omegaconf import DictConfig
from datetime import datetime
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error

from torch.optim.lr_scheduler import StepLR
from src.utils import torch_fix_seed, fetch_sample_data, ipm_scores, get_x_t, get_environments, ihdp_data_prep
from src.utils import TD_DataSet
from src.cfr import CFR
from sklearn.model_selection import train_test_split


import random
import shutil
import sys
import operator
from numbers import Number
from collections import OrderedDict, Counter
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from random import randrange

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim
import argparse
from pathlib import Path

@hydra.main(config_path="configs", config_name="experiments.yaml", version_base=None)
def main(cfg: DictConfig):
    x_t_name = 'head-circumference'
    number_environments = 3
    ihdp_data_compressed, variable_dict, true_ate = ihdp_data_prep()

    # print(ihdp_data_compressed.shape, ihdp_data_compressed.columns)

    x_t = get_x_t(ihdp_data_compressed, variable_dict, x_t_name)
    ihdp_data_compressed['e_1'] = get_environments(x_t,ihdp_data_compressed['treatment'].to_numpy().reshape(-1,1),1, number_environments)

    ihdp_control = ihdp_data_compressed[ihdp_data_compressed['treatment'] == 0] 
    ihdp_treated = ihdp_data_compressed[ihdp_data_compressed['treatment'] == 1]

    treatments_control = ihdp_control['treatment']
    treatments_treated = ihdp_treated['treatment']      
    
    ite_control = ihdp_control['ite']
    ite_treated = ihdp_treated['ite']

    torch_data_control = ihdp_control.drop(columns = ['y_cfactual', 'y_factual', 'ite', 'treatment']) 
    torch_data_control = torch_data_control.drop(columns = [variable_dict[x_t_name]])

    torch_data_treated = ihdp_treated.drop(columns = ['y_cfactual', 'y_factual', 'ite', 'treatment'])
    torch_data_treated = torch_data_treated.drop(columns = [variable_dict[x_t_name]])

    num_features_control = torch_data_control.shape[1]
    torch_labels_control = ihdp_control['y_factual']

    num_features_treated = torch_data_treated.shape[1]
    torch_labels_treated = ihdp_treated['y_factual']

    for group in ['control', 'treated']:
        locals()[f'X_train_{group}'], locals()[f'X_test_{group}'], locals()[f'y_train_{group}'], locals()[f'y_test_{group}'], locals()[f't_train_{group}'], locals()[f't_test_{group}'], locals()[f'ite_train_{group}'], locals()[f'ite_test_{group}'] = train_test_split(
                    locals()[f'torch_data_{group}'],
                    locals()[f'torch_labels_{group}'],
                    locals()[f'treatments_{group}'],
                    locals()[f'ite_{group}'],
                    test_size=0.15)

        locals()[f'X_train_{group}'], locals()[f'X_test_{group}'] = np.array(locals()[f'X_train_{group}']), np.array(locals()[f'X_test_{group}'])
        locals()[f'y_train_{group}'], locals()[f'y_test_{group}'] = np.array(locals()[f'y_train_{group}']), np.array(locals()[f'y_test_{group}'])
        locals()[f't_train_{group}'], locals()[f't_test_{group}'] = np.array(locals()[f't_train_{group}']), np.array(locals()[f't_test_{group}'])
        locals()[f'ite_train_{group}'], locals()[f'ite_test_{group}'] = np.array(locals()[f'ite_train_{group}']), np.array(locals()[f'ite_test_{group}'])
        
        locals()[f'X_train_{group}'], locals()[f'X_test_{group}'] = torch.FloatTensor(locals()[f'X_train_{group}']), torch.FloatTensor(locals()[f'X_test_{group}'])
        locals()[f'y_train_{group}'], locals()[f'y_test_{group}'] = torch.FloatTensor(locals()[f'y_train_{group}']).unsqueeze(1), torch.FloatTensor(locals()[f'y_test_{group}']).unsqueeze(1)
        locals()[f't_train_{group}'], locals()[f't_test_{group}'] = torch.FloatTensor(locals()[f't_train_{group}']).unsqueeze(1), torch.FloatTensor(locals()[f't_test_{group}']).unsqueeze(1)
        locals()[f'ite_train_{group}'], locals()[f'ite_test_{group}'] = torch.FloatTensor(locals()[f'ite_train_{group}']).unsqueeze(1), torch.FloatTensor(locals()[f'ite_test_{group}']).unsqueeze(1)
        
        # print(locals()[f'X_train_{group}'].size(), locals()[f'X_test_{group}'].size())
        # print(locals()[f'y_train_{group}'].size(), locals()[f'y_test_{group}'].size())
        # print(locals()[f't_train_{group}'].size(), locals()[f't_test_{group}'].size()) 
        
        dataset = TD_DataSet(locals()[f'X_train_{group}'], locals()[f'y_train_{group}'], locals()[f't_train_{group}'], locals()[f'ite_train_{group}'])
        locals()[f'dataloader_{group}'] = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, drop_last=True)

        model = CFR(in_dim=locals()[f'X_train_{group}'].shape[1], out_dim=1, cfg=cfg)
        opt = getattr(optim, 'Adam')

        for i in range(10):
            locals()[f'within_result_{group}'], locals()[f'outof_result_{group}'] = model.train_tlearner(
                    locals()[f'dataloader_{group}'], locals()[f'X_train_{group}'], locals()[f'y_train_{group}'], locals()[f't_train_{group}'], locals()[f'ite_train_{group}'], locals()[f'X_test_{group}'], locals()[f'y_test_{group}'], locals()[f't_test_{group}'], locals()[f'ite_test_{group}'], opt, i)

            

if __name__ == '__main__':
    main()