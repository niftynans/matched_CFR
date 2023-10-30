import hydra
import os
import argparse

from omegaconf import DictConfig
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
from datetime import datetime
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error

from torch.optim.lr_scheduler import StepLR
import mlflow
from mlflow import log_metric, log_param, log_artifacts

from src.utils import torch_fix_seed, fetch_sample_data, ipm_scores
from src.cfr import CFR

import torch
import os
import random
import shutil
import sys
import operator
from numbers import Number
from collections import OrderedDict, Counter
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec


import torch
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='Treatment Effect Estimation using Gradient Matching.')
parser.add_argument('--dataset', type=str, help="Name of dataset: ihdp, jobs", default='ihdp')
parser.add_argument('--algorithm', type=str, help="Training scheme: fish, erm", default='erm')
args = parser.parse_args()

@hydra.main(config_path="configs", config_name="experiments.yaml", version_base=None)
def run_experiment(cfg: DictConfig):

    # start new run
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment("CFR experiments")

    logger = getLogger("run DFR")
    logger.setLevel(DEBUG)
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(handler_format)
    try:
        os.mkdir('logs')
    except:
        pass
    file_handler = FileHandler(hydra.utils.get_original_cwd() + "/logs/" + "cfr" + "-" + "{:%Y-%m-%d-%H:%M:%S}.log".format(datetime.now()), "a")
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug(f"Start process...")

    with mlflow.start_run():

        torch_fix_seed()
        log_param("alpha", cfg["alpha"])
        log_param("split_outnet", cfg["split_outnet"])
        log_param("train_test_random", cfg["random_state"])
        try:
            os.mkdir('data')
        except:
            pass
        (
            dataloader,
            X_train,
            y_train,
            t_train,
            ite_train,
            X_test,
            y_test,
            t_test,
            ite_test
        ) = fetch_sample_data(
            random_state=cfg["random_state"], test_size=0.25, StandardScaler=cfg["StandardScaler"], data_path=hydra.utils.get_original_cwd() + "/data/sample_data.csv", dataset= args.dataset
        )
    
        model = CFR(in_dim=X_train.shape[1], out_dim=1, cfg=cfg)
        opt = getattr(optim, 'Adam')

        if args.algorithm == 'fish':
            within_result, outof_result, train_mse, ipm_result = model.train_fish(
                dataloader, X_train, y_train, t_train, X_test, y_test, t_test, logger, opt, ite_train, ite_test)
                
        else:
            within_result, outof_result, train_mse, ipm_result = model.fit(
            dataloader, X_train, y_train, t_train, X_test, y_test, t_test, logger, ite_train, ite_test)
    
        
        within_ipm = ipm_scores(model, X_train, t_train, sig=0.1)
        outof_ipm = ipm_scores(model, X_test, t_test, sig=0.1)

        
        # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        # ax1.bar(range(len(within_result)), list(within_result.values()), align='center')
        # ax1.set_xticks(range(len(within_result)), list(within_result.keys()))
        # for i, v in enumerate(within_result.values()):
        #     ax1.text(range(len(within_result))[i] - 0.25, v + 0.01, str(v))
        # ax1.set_title('within_sample')
        
        # ax2.bar(range(len(outof_result)), list(outof_result.values()), align='center')
        # ax2.set_xticks(range(len(outof_result)), list(outof_result.keys()))
        # for i, v in enumerate(outof_result.values()):
        #     ax2.text(range(len(outof_result))[i] - 0.25, v + 0.01, str(v))
        # ax2.set_title('outof_sample')
        
        # # model-fish-confounding-x_t-name
        # try:
        #     os.mkdir('result_imgs')
        # except:
        #     pass
        # fig.savefig('result_imgs/CFRNet_fish_confounding_age.png')
        
        log_param("within_IPM_pre", within_ipm["ipm_lin_before"])
        log_param("outof_IPM_pre", outof_ipm["ipm_lin_before"])

        log_metric("within_ATT", within_result["ATT"])
        log_metric("within_ATTerror", np.abs(within_result["ATT"] - 1676.3426))
        log_metric("within_RMSE", within_result["RMSE"])
        log_metric("within_IPM", within_ipm["ipm_lin"])

        log_metric("outof_ATT", outof_result["ATT"])
        log_metric("outof_ATTerror", np.abs(outof_result["ATT"] - 1676.3426))
        log_metric("outof_RMSE", outof_result["RMSE"])
        log_metric("outof_IPM", outof_ipm["ipm_lin"])            

if __name__ == "__main__":
    run_experiment()