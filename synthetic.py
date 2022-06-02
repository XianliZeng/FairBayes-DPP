import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

import random

import pandas as pd



from utils_synthetic import generatedata as gd
from utils_synthetic import measure_from_Yhat

from utils import threshold_pp


pa = 0.3
py1set = 0.6
py0 = 0.2
n_train= 50000
n_test = 5000
c=0.5
p=2
mu11 = np.array([1,1])
mu10 = np.array([1,-1])
mu01 = np.array([-1,1])
mu00 = np.array([-1,-1])
sigma =2

n_seed = 100
acc_PP = pd.DataFrame()
acc_PP_baseline = pd.DataFrame()
method1='FairBayes-DPP'
method2='Unconstrained'
for seed in range(n_seed):
    seed=seed*5
    print(f'seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)





    #generate training and test data
    [X_train,A_train,Y_train,] = gd(mu11,mu10,mu01,mu00,pa,py1,py0,n_train,p,sigma)
    X1_train, Y1_train = X_train[A_train==1], Y_train[A_train==1]
    X0_train, Y0_train = X_train[A_train==0], Y_train[A_train==0]


    [X_test,A_test,Y_test] = gd(mu11,mu10,mu01,mu00,pa,py1,py0,n_test,p,sigma)
    X1_test, Y1_test = X_test[A_test==1], Y_test[A_test==1]
    X0_test, Y0_test = X_test[A_test==0], Y_test[A_test==0]


    #logistic regression for subgroups
    model1 = LogisticRegression(solver='liblinear', random_state=0).fit(X1_train,Y1_train)
    model0 = LogisticRegression(solver='liblinear', random_state=0).fit(X0_train,Y0_train)




# calculate group-wise conditional probabilities
    eta1_train = model1.predict_proba(X1_train)[:, 1]
    eta0_train = model0.predict_proba(X0_train)[:, 1]





    Y1_train=np.array(Y1_train)
    Y0_train=np.array(Y0_train)
    Y1_test=np.array(Y1_test)
    Y0_test=np.array(Y0_test)

    # Calculate fair thresholds
    [t1star, t0star] = threshold_pp(eta1_train, eta0_train, Y1_train, Y0_train)
    eta1_test = model1.predict_proba(X1_test)[:, 1]
    eta0_test = model0.predict_proba(X0_test)[:, 1]
    Yhat1=(eta1_test>=t1star)
    Yhat0=(eta0_test>=t0star)
    Yhat1_baseline=(eta1_test>=c)
    Yhat0_baseline=(eta0_test>=c)


    result_temp_fair = measure_from_Yhat(Yhat1, Yhat0, Y1_test, Y0_test)
    result_temp_baseline = measure_from_Yhat(Yhat1_baseline, Yhat0_baseline, Y1_test, Y0_test)
    result_temp_fair['seed']=seed
    result_temp_baseline['seed']=seed

    result_temp_fair['method']=method1
    result_temp_baseline['method']=method2

    acc_PP=acc_PP.append(result_temp_fair)
    acc_PP_baseline=acc_PP_baseline.append(result_temp_baseline)



