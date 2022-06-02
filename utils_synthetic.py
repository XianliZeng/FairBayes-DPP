
import numpy as np
import pandas as pd
def generatedata(mu11,mu10,mu01,mu00,pa,py1,py0,n,p,sigma):
    p11 = pa * py1
    p10 = pa * (1 - py1)
    p01 = (1 - pa) * py0
    p00 = (1 - pa) * (1 - py0)
    pvec = [p11, p10, p01, p00]

    dic_mu = {1: mu11, 2: mu10, 3: mu01, 4: mu00}
    dic_ay = {1: [1,1], 2: [1,0], 3: [0,1], 4: [0,0]}


    idx = np.random.choice(list(dic_mu.keys()), n, replace=True, p=pvec)
    data_ay = np.array([dic_ay[key] for key in idx])
    data_mu = np.array([dic_mu[key] for key in idx])

    X = sigma * np.random.randn(n, p)
    X = X +  data_mu
    A = data_ay[:, 0]
    Y = data_ay[:, 1]

    return [X,A,Y]





def measure_from_Yhat(Yhat1, Yhat0, Y1, Y0):
    if (Yhat1 == 1).mean()==0:
        p1 = 1
    else:
        p1 = np.mean(Y1[Yhat1 == 1])

    if (Yhat0 == 1).mean()==0:
        p0 = 1
    else:
        p0 = np.mean(Y0[Yhat0 == 1])
    n_test = len(Y1)+ len(Y0)
    DPP_LR = np.abs( p1 - p0 )
    acc_LR = (np.sum(Y1==Yhat1) + np.sum(Y0==Yhat0))/n_test

    data_LR = [ DPP_LR,acc_LR]

    columns = ['DPP','acc']
    result_temp = pd.DataFrame([data_LR], columns=columns)

    return result_temp