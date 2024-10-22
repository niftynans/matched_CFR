import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from src.utils import ihdp_data_prep, get_cattaneo_compressed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import mean_squared_error
import sys

def get_score_slearner(s_learner, X_train, X_test, y_train, y_test, t_train, t_test, ite_train = None, ite_test = None):
    print("S-Learner")
    N = len(X_test)
    s_learner.fit(X_train, y_train)
    
    # MSE
    _ypred = s_learner.predict(X_test)
    mse = mean_squared_error(y_test, _ypred)
    
    t_idx = np.where(t_test == 1)[0]
    c_idx = np.where(t_test == 0)[0]
    _t0 = np.array([0 for _ in range(N)]).reshape([-1,1])
    _t1 = np.array([1 for _ in range(N)]).reshape([-1,1])

    X_test['treatment'] = _t1
    thresh_1 = s_learner.predict(X_test)
    X_test['treatment'] = _t0
    thresh_0 = s_learner.predict(X_test)
    
    thresh = thresh_1 - thresh_0
    
    if ite_test is not None:
        pehe = np.sqrt(np.mean(np.square(thresh - ite_test)))
        print('PEHE: ', pehe)
    
    X_test['treatment'] = _t0
    _cate_t = y_test - s_learner.predict(X_test)
    _cate_t.reset_index(drop = True, inplace = True)
    X_test['treatment'] = _t1
    _cate_c = s_learner.predict(X_test) - y_test
    _cate_c.reset_index(drop = True, inplace = True)
    _cate = np.append(_cate_c[c_idx], _cate_t[t_idx])

    if ite_test is not None:
        ate_error = np.mean(np.abs(4.016066896118338 - _cate))
        print("E_ATE: ", ate_error)
    
    if ite_test is None:
        threshold = np.mean(thresh)
        _att = np.mean(_cate_t[t_idx])
        _atc = np.mean(_cate_c[c_idx])
        print('E_ATT: ',np.abs((_att - threshold) / (_att+ threshold)))
        print('E_ATC: ',np.abs((_atc - threshold) / (_atc + threshold)))
        
def get_score_tlearner(m0, m1, X_train, X_test, y_train, y_test, t_train, t_test, ite_train = None, ite_test = None):
    print("T-Learner")
    N = len(X_test)
    
    m1.fit(X_train[X_train['treatment'] == 1], y_train[X_train['treatment'] == 1])
    m0.fit(X_train[X_train['treatment'] == 0], y_train[X_train['treatment'] == 0])
            
    t_idx = np.where(t_test == 1)[0]
    c_idx = np.where(t_test == 0)[0]
    
    _t0 = np.array([0 for _ in range(N)]).reshape([-1,1])
    _t1 = np.array([1 for _ in range(N)]).reshape([-1,1])

    thresh_1 = m1.predict(X_test)
    thresh_0 = m0.predict(X_test)
    
    thresh = thresh_1 - thresh_0
    
    if ite_test is not None:
        pehe = np.sqrt(np.mean(np.square(thresh - ite_test)))
        print('PEHE: ', pehe)
    
    _cate_t = y_test - m1.predict(X_test)
    _cate_c = m0.predict(X_test) - y_test
    _cate_t.reset_index(drop = True, inplace = True)
    _cate_c.reset_index(drop = True, inplace = True)
    _cate = np.append(_cate_c[c_idx], _cate_t[t_idx])
    if ite_test is not None:
        ate = np.mean(_cate)
        ate_error = np.abs(4.016066896118338 - ate)
        print("E_ATE: ", ate_error)

    if ite_test is None:
        threshold = np.mean(thresh)
        _att = np.mean(_cate_t[t_idx])
        _atc = np.mean(_cate_c[c_idx])
        print('E_ATT: ', np.abs((_att - threshold) / (_att+ threshold)))
        print('E_ATC: ', np.abs((_atc - threshold) / (_atc + threshold)))

def main(slearner = True, tlearner = True, dataset = 'cattaneo'):
    if dataset == 'ihdp':
        data, variable_dict, true_ate = ihdp_data_prep()
        ite = data['ite']
        Y = data['y_factual']
        T = data['treatment']
        X = data.drop(columns = ['y_cfactual', 'y_factual', 'ite', variable_dict['twin'], variable_dict['married'], 
                                                            variable_dict['edu-left-hs'] , variable_dict['edu-hs'],variable_dict['edu-sc'],variable_dict['cig'],
                                                            variable_dict['first-born'], variable_dict['alcohol'], variable_dict['working'],
                                                            variable_dict['ark'],variable_dict['ein'],variable_dict['har'],variable_dict['mia'],
                                                            variable_dict['pen'],variable_dict['tex'], variable_dict['was']])
        
        X_train, X_test, y_train, y_test, t_train, t_test, ite_train, ite_test = train_test_split( X, Y, T, ite, test_size=0.2)
        
        s_learner = LinearRegression()
        # get_score_slearner(s_learner, X_train, X_test, y_train, y_test, t_train, t_test, ite_train, ite_test)
        m0 = LinearRegression()
        m1 = LinearRegression()
        get_score_tlearner(m0, m1, X_train, X_test, y_train, y_test, t_train, t_test, ite_train, ite_test)
        sys.exit()

    elif dataset == 'jobs':
        data_path="data/sample_data.csv"
        dataset='jobs'
        data = pd.read_csv(data_path)
        data = data.rename(columns={"treat": "treatment"})
        Y = data['re78']
        T = data['treatment']
        X = data.drop(["re78"], axis=1)
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split( X, Y, T, test_size=0.2)


    else:
        data, variable_dict = get_cattaneo_compressed()
        Y = data['y']
        T = data['treatment']
        X = data.drop(["y"], axis=1)
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split( X, Y, T, test_size=0.2)

    s_learner = LinearRegression()
    # get_score_slearner(s_learner, X_train, X_test, y_train, y_test, t_train, t_test)
    m0 = LinearRegression()
    m1 = LinearRegression()
    get_score_tlearner(m0, m1, X_train, X_test, y_train, y_test, t_train, t_test)
    sys.exit()            
    
if __name__ == '__main__':
    main()