import numpy as np
import pandas as pd

import numpy as np

###calculate thresholds to satisfies DPP constraint######



def balance_ppv(eta_base, eta, Ybase, Y,tbase ,tmin,tmax):
    if (eta_base >= tbase).mean() == 0:
        s=1
    else:
        s= np.mean(Ybase[eta_base>= tbase])

    for i in range(20):
        t=(tmin+tmax)/2
        if (eta >= t).mean() == 0:
            sc = 1
        else:
            sc =  np.mean(Y[eta>= t])
        if sc>s:
            tmax=t
        else:
            tmin=t
    return (tmax+tmin)/2


def threshold_pp(eta1, eta0, Y1, Y0):
    tmax1=np.max(eta1)
    tmax0=np.max(eta0)

    datasize=len(eta1)+len(eta0)
    if (eta1 >= 1/2).mean() == 0:
        s1 = 1
    else:
        s1 = np.mean(Y1[eta1>= 1/2])

    if (eta0 >= 1/2).mean() == 0:
        s0 = 1
    else:
        s0 = np.mean(Y0[eta0 >= 1 / 2])


    if s1>s0:
        t1max=0.5
        t1min=balance_ppv(eta0, eta1, Y0, Y1,1/2 ,0.001,1/2)
        t0min=0.5
        t0max=balance_ppv(eta1, eta0, Y1, Y0,1/2 ,0.5,tmax0)
        t1set = np.arange(t1min, t1max, 0.001)
        lent = len(t1set)
        t0set = [balance_ppv(eta1, eta0, Y1, Y0, t1, t0min, t0max) for t1 in t1set]
        accset = [(((eta1 >= t1set[s]) == Y1).sum() + ((eta0 >= t0set[s]) == Y0).sum()) / datasize for s in range(lent)]
        accset=np.array(accset)
        index = np.argmax(accset)
        t1star=t1set[index]
        t0star=t0set[index]
    else:
        t1min=0.5
        t1max=balance_ppv(eta0, eta1, Y0, Y1,1/2 ,1/2,tmax1)
        t0min = balance_ppv(eta1, eta0, Y1, Y0, 1 / 2, 0, 0.5)
        t0max = 0.5
        t1set = np.arange(t1min, t1max, 0.001)
        lent = len(t1set)
        t0set = [balance_ppv(eta1, eta0, Y1, Y0, t1, t0min, t0max,) for t1 in t1set]
        accset = [(((eta1 >= t1set[s]) == Y1).sum() + ((eta0 >= t0set[s]) == Y0).sum()) / datasize for s in range(lent)]
        t0set=np.array(t0set)
        accset=np.array(accset)
        index = np.argmax(accset)
        t1star=t1set[index]
        t0star=t0set[index]




    return [t1star,t0star]







