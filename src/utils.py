import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import numpy as np
import torch
from random import randrange
import pyreadr
from scipy.special import expit, softmax
from sklearn.preprocessing import MinMaxScaler    
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error


def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class TD_DataSet(Dataset):
    def __init__(self, x, y, z, batch_size = 16):
        self.x = x
        self.y = y
        self.z = z
        domains = x[:,-1].clone().detach()
        indices = []
        for i in range(x.size()[0]):
            indices.append(i)  
        indices = torch.tensor(indices)
        self.domain_indices = [indices[domains == d] for d in domains.unique()]
        self.domains = domains
        self.targets = y
        self.train_data = x
        self.batch_size = 16

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.get_input(i) for i in batch_index]), \
               self.targets[batch_index], self.z[batch_index],self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        return self.train_data[idx]

    def eval(self, ypreds, ys, metas):
        loss = mean_squared_error(ys, ypreds)
        test_val = [
            {'mse_loss': loss}
        ]
        return test_val

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index], self.z[index, :]
    

def ndarray_to_tensor(x):
        x = torch.tensor(x).float()
        return x

def fetch_sample_data(random_state=0, test_size=0.15, StandardScaler=False, data_path="data/sample_data.csv", dataset='jobs'):
    if dataset == 'jobs':
        if os.path.isfile(data_path):
            df = pd.read_csv(data_path)
            print(df.columns)
            a = randrange(1, df.shape[1]/5)
            print(a)
            feats = []
            for i in range(a):
                feats.append(randrange(df.shape[1]))
            feats = list(set(feats))
            print(df.columns[feats])
            # df = df.drop(columns=df.columns[feats])

        else:
            RCT_DATA = "http://www.nber.org/~rdehejia/data/nsw_dw.dta"
            CPS_DATA = "http://www.nber.org/~rdehejia/data/cps_controls3.dta"  

            df = pd.concat(
                [
                    pd.read_stata(RCT_DATA).query("treat>0"),  
                    pd.read_stata(CPS_DATA),  
                ]
            ).reset_index(drop=True)

            del df["data_id"]

            df["treat"] = df["treat"].astype(int)
            df.to_csv(data_path, index=False)

        if StandardScaler:
            features_cols = [col for col in df.columns if col not in ["treat", "re78"]]
            ss = preprocessing.StandardScaler()
            df_std = pd.DataFrame(ss.fit_transform(df[features_cols]), columns=features_cols)
            df_std = pd.concat([df[["treat", "re78"]], df_std], axis=1)
            df = df_std.copy()
        
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            df.drop(["re78", "treat"], axis=1),
            df[["re78"]],
            df[["treat"]],
            random_state=random_state,
            test_size=test_size,
        )
        X_train = torch.FloatTensor(X_train.to_numpy())
        y_train = torch.FloatTensor(y_train.to_numpy())
        t_train = torch.FloatTensor(t_train.to_numpy())

        X_test = torch.FloatTensor(X_test.to_numpy())
        y_test = torch.FloatTensor(y_test.to_numpy())
        t_test = torch.FloatTensor(t_test.to_numpy())
        
        # print(X_train.size(), y_train.size(), t_train.size())
        # print(X_test.size(), y_test.size(), t_test.size())

    else:
        confounding = False
        x_t_name = 'birth-weight'
        number_environments = 3
        ihdp_data_compressed, variable_dict, true_ate = ihdp_data_prep()
        x_t = get_x_t(ihdp_data_compressed, variable_dict, x_t_name)
        ihdp_data_compressed['e_1'] = get_environments(x_t,ihdp_data_compressed['treatment'].to_numpy().reshape(-1,1),1, number_environments)
        ite = ihdp_data_compressed['ite']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        treatments = ihdp_data_compressed['treatment']        
        torch_data = ihdp_data_compressed.drop(columns = ['y_cfactual', 'y_factual', 'ite', 'treatment'])
        torch_data = torch_data.drop(columns = [variable_dict[x_t_name]])
        if confounding:
            a = randrange(1, int(torch_data.shape[1]/5))
            feats = []
            for i in range(a):
                feats.append(randrange(torch_data.shape[1]))
            feats = list(set(feats))
            torch_data = torch_data.drop(columns=torch_data.columns[feats])

        num_features = torch_data.shape[1]
        torch_labels = ihdp_data_compressed['y_factual']
        
        X_train, X_test, y_train, y_test, t_train, t_test, ite_train, ite_test = train_test_split(
            torch_data,
            torch_labels,
            treatments,
            ite,
            random_state=random_state,
            test_size=test_size)
        
        
        # scaler = MinMaxScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        t_train, t_test = np.array(t_train), np.array(t_test)
        X_test, y_test = np.array(X_test), np.array(y_test)
        ite_train, ite_test = np.array(ite_train), np.array(ite_test)
        
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        t_train = torch.FloatTensor(t_train).unsqueeze(1)
        ite_train = torch.FloatTensor(ite_train).unsqueeze(1)

        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        t_test = torch.FloatTensor(t_test).unsqueeze(1)
        ite_test = torch.FloatTensor(ite_test).unsqueeze(1)

        # print(X_train.size(), y_train.size(), t_train.size())
        # print(X_test.size(), y_test.size(), t_test.size())
        
    dataset = TD_DataSet(X_train, y_train, t_train, ite_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    return  dataloader, X_train, y_train, t_train, ite_train, X_test, y_test, t_test, ite_test


def mmd_rbf(Xt, Xc, p, sig=0.1):
    sig = torch.tensor(sig)
    Kcc = torch.exp(-torch.cdist(Xc, Xc, 2.0001) / torch.sqrt(sig))
    Kct = torch.exp(-torch.cdist(Xc, Xt, 2.0001) / torch.sqrt(sig))
    Ktt = torch.exp(-torch.cdist(Xt, Xt, 2.0001) / torch.sqrt(sig))

    m = Xc.shape[0]
    n = Xt.shape[0]

    mmd = (1 - p) ** 2 / (m *(m-1)) * (Kcc.sum() - m)
    mmd += p ** 2 / (n * (n-1)) * (Ktt.sum() - n)
    mmd -= 2 * p * (1 - p) / (m * n) * Kct.sum()
    mmd *= 4

    return mmd

def mmd_lin(Xt, Xc, p):
    mean_treated = torch.mean(Xt)
    mean_control = torch.mean(Xc)
    
    mmd = torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control).sum()

    return mmd


def ipm_scores(model, X, _t, sig=0.1):
    _t_id = np.where((_t.cpu().detach().numpy() == 1).all(axis=1))[0]
    _c_id = np.where((_t.cpu().detach().numpy() == 0).all(axis=1))[0]
    x_rep = model.repnet(X)
    ipm_lin = mmd_lin(
                    x_rep[_t_id],
                    x_rep[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id))
    )
    ipm_rbf = mmd_rbf(
                    x_rep[_t_id],
                    x_rep[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id)),
                    sig=sig,
                )
    ipm_lin_pre = mmd_lin(
                    X[_t_id],
                    X[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id))
    )
    ipm_rbf_pre = mmd_rbf(
                    X[_t_id],
                    X[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id)),
                    sig=sig,
                )
    return {
        "ipm_lin": np.float64(ipm_lin.cpu().detach().numpy()),
        "ipm_rbf": np.float64(ipm_rbf.cpu().detach().numpy()),
        "ipm_lin_before": np.float64(ipm_lin_pre.cpu().detach().numpy()),
        "ipm_rbf_before": np.float64(ipm_rbf_pre.cpu().detach().numpy()),
    }
    
def ihdp_data_prep():
    
    """
    Function which loads the IHDP dataset, and returns it in a compact form, along with a variable dict, and the true ATE.
    """
    
    # data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
    data = pd.read_csv('data/ihdp_npci_1.csv', header=None)
    col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,]
    for i in range(1,26):
        col.append(str(i))
    data.columns = col

    true_ate = np.mean(data['mu1']-data['mu0'])

    ihdp_data = pyreadr.read_r('data/ihdp.RData')['ihdp']
    ihdp_data = ihdp_data[(ihdp_data['treat'] != 1) | (ihdp_data['momwhite'] != 0)].reset_index(drop=True)
    ihdp_data = ihdp_data.drop(['momwhite', 'momblack', 'momhisp'], axis=1)

    ihdp_data_compressed = ihdp_data.copy()
    cols_to_norm = ['bw','b.head','preterm','birth.o','nnhealth','momage']
    ihdp_data_compressed[cols_to_norm] = ihdp_data_compressed[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))
    ihdp_data_compressed.columns = ['treatment','birth-weight','head-circumference','pre-term','birth-order','neonatal','age','sex','twin','married','edu-left-hs','edu-hs','edu-sc','cig','first-born','alcohol','drugs','working','prenatal','ark','ein','har','mia','pen','tex','was']
    ihdp_data_compressed['y_factual'] = data['y_factual']
    ihdp_data_compressed['y_cfactual'] = data['y_cfactual']
    ihdp_data_compressed['ite'] = (ihdp_data_compressed['y_factual'] - ihdp_data_compressed['y_cfactual'])*(2*ihdp_data_compressed['treatment']-1)
    cols_list = ihdp_data_compressed.columns.tolist()
    ihdp_data_compressed = ihdp_data_compressed[cols_list[1:26] + cols_list[0:1] + cols_list[26:]]
    variable_dict = {'birth-weight':"1",
                    'head-circumference':"2",
                    'pre-term': "3",
                    'birth-order': "4",
                    'neonatal':"5",
                    'age':"6",
                    'sex':"7",
                    'twin':"8",
                    'married':"9",
                    'edu-left-hs':"10",
                    'edu-hs':"10",
                    'edu-sc':"10",
                    'cig':"11",
                    'first-born':"12",
                    'alcohol':"13",
                    'drugs':"14",
                    'working':"15",
                    'prenatal':"16",
                    'ark':"17",
                    'ein':"17",
                    'har':"17",
                    'mia':"17",
                    'pen':"17",
                    'tex':"17",
                    'was':"17"}
    ihdp_data_compressed.columns = ["1","2","3","4","5","6","7","8","9","10","10","10","11","12","13","14","15","16","17","17","17","17","17","17","17","treatment", "y_factual", "y_cfactual","ite"] 
    return ihdp_data_compressed, variable_dict, true_ate


def get_x_t(ihdp_data_compressed, variable_dict, x_t_name):
    """ 
    Function to retrieve x_t (The variable to split the dataset across environments)
    """
    x_t = ihdp_data_compressed[variable_dict[x_t_name]].to_numpy().reshape(-1,1)
    return x_t

def get_environments(x_t,t,use_t_in_e,number_environments):
    use_t_in_e = 1
    if use_t_in_e == 0:
        theta = np.concatenate((np.random.uniform(1.0,2.0,(1,1)), np.zeros((1,1)), np.random.uniform(-1.0,-2.0,(1,1))), axis = 1)
        probabilites = softmax(np.dot(x_t, theta)-np.mean(np.dot(x_t, theta), axis = 0), axis = 1)
    else:
        theta = np.concatenate((np.random.uniform(1.0,2.0,(2,1)), np.zeros((2,1)), np.random.uniform(-1.0,-2.0,(2,1))), axis = 1)
        probabilites = softmax(np.dot(np.concatenate((x_t, t), axis = 1), theta)-np.mean(np.dot(np.concatenate((x_t, t), axis = 1), theta),  axis = 0 ),  axis = 1)

    e = np.zeros(x_t.shape, dtype=int)
    for i,probability in enumerate(probabilites):
        e[i,:] = np.random.choice(np.arange(number_environments), p = probability)
    return e

