from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision
from medmnist import ChestMNIST
from PIL import Image, ImageDraw
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def load_imgs():
    covariates = []
    treatments = []
    outcomes = []
    path = 'data/causal_xray/test/'
    for file in os.listdir(path):
        if '_z' in file:        
            img = Image.open(path + file).convert('L')
            img = np.stack((img,)*1, axis=-1)
            covariates.append(img)
            
            file_t = file.replace('_z', '_x')
            img = Image.open(path + file_t).convert('L')
            img = np.stack((img,)*1, axis=-1)
            treatments.append(img)

            file_y = file.replace('_z', '_y')
            img = Image.open(path + file_y).convert('L')
            img = np.stack((img,)*1, axis=-1)
            outcomes.append(img)

    return np.array(covariates), np.array(treatments), np.array(outcomes)

def main():
    covariates, treatments, outcomes = load_imgs()    
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(covariates, outcomes, treatments, test_size = 0.1)
    
    X_train = X_train.reshape(len(X_train), 1, 28, 28)
    X_train  = torch.from_numpy(X_train)
    y_train = y_train.reshape(len(y_train), 1, 28, 28)
    y_train = torch.from_numpy(y_train)
    t_train = t_train.reshape(len(t_train), 1, 28, 28)
    t_train = torch.from_numpy(t_train)

if __name__ == '__main__':
    main()