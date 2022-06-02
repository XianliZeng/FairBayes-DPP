import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from utils import threshold_pp
import sys

def FairBayes_DPP(dataset,dataset_name,net, optimizer, device, n_epochs=200, batch_size=2048, seed=0):

    # Retrieve train/test splitted pytorch tensors for index=split
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_val_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train_val, Y_train_val, Z_train_val, XZ_train_val  = train_val_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors
    X_all = torch.cat((X_train_val, X_test), 0)
    Y_all = torch.cat((Y_train_val, Y_test), 0)
    Z_all = torch.cat((Z_train_val, Z_test), 0)
    XZ_all = torch.cat((XZ_train_val, XZ_test), 0)


    datasize=len(X_all)
    train_size=int(datasize*0.7)
    val_size=int(datasize*0.5)
    test_size=int(datasize*0.3)
    index=np.arange(0,datasize,1)
    train_index_sum=np.random.choice(index,train_size)
    val_index_sum=np.random.choice(index,val_size)
    test_index_sum=np.random.choice(index,test_size)



    Y_train = Y_all[train_index_sum]
    Y_val = Y_all[val_index_sum]
    Y_test = Y_all[test_index_sum]


    Z_train = Z_all[train_index_sum]
    Z_train_np = Z_train.detach().cpu().numpy()
    Z_list = sorted(list(set(Z_train_np)))
    for z in Z_list:
        if (Z_train_np==z).sum()==0:
            print('At least one sensitive group has no data point')
            sys.exit()
    Z_val = Z_all[val_index_sum]
    Z_test = Z_all[test_index_sum]
    test_size=len(Z_test)


    XZ_train = XZ_all[train_index_sum]
    XZ_val = XZ_all[val_index_sum]
    XZ_test = XZ_all[test_index_sum]

    XZ_val_att1, Y_val_att1 = XZ_val[Z_val == 1], Y_val[Z_val == 1]
    XZ_val_att0, Y_val_att0 = XZ_val[Z_val == 0], Y_val[Z_val == 0]
    XZ_test_att1, Y_test_att1 = XZ_test[Z_test == 1], Y_test[Z_test == 1]
    XZ_test_att0, Y_test_att0 = XZ_test[Z_test == 0], Y_test[Z_test == 0]


    Y_val_att1_np=Y_val_att1.clone().cpu().detach().numpy()
    Y_val_att0_np=Y_val_att0.clone().cpu().detach().numpy()

    Y_val_np=Y_val.clone().cpu().detach().numpy()

    Y_test_att1_np = Y_test_att1.clone().cpu().detach().numpy()
    Y_test_att0_np = Y_test_att0.clone().cpu().detach().numpy()

    custom_dataset = CustomDataset(XZ_train, Y_train)
    data_loader= DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    loss_function  = nn.BCELoss()

    costs = []
    total_train_step= 0



    for epoch in range(n_epochs):
        net.train()

        for i,(x,y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            Yhat = net(x)
            loss = loss_function(Yhat.squeeze(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            costs.append(loss.item())

            total_train_step += 1
            if (i + 1) % 10 == 0 or (i + 1) == batch_size:
                print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(epoch+1, n_epochs,
                                                                          i+1, len(data_loader),
                                                                          loss.item()), end='\r')



########choose the model with best performance on validation set###########
        net.eval()
        with torch.no_grad():

            output_val = net(XZ_val).squeeze().detach().cpu().numpy()

            Ytilde_val = (output_val >= 0.5).astype(np.float32)

            accuracy = (Ytilde_val == Y_val_np).astype(np.float32).mean()


            if epoch==0:
                accuracy_max=accuracy
                bestnet_acc_stat_dict=net.state_dict()


            if accuracy > accuracy_max:
                accuracy_max=accuracy
                bestnet_acc_stat_dict=net.state_dict()


#########Calculate thresholds for fair Bayes-optimal Classifier###########
    net.load_state_dict(bestnet_acc_stat_dict)


    eta1_val = net(XZ_val_att1).squeeze().detach().cpu().numpy()
    eta0_val = net(XZ_val_att0).squeeze().detach().cpu().numpy()
    eta1_test = net(XZ_test_att1).squeeze().detach().cpu().numpy()
    eta0_test = net(XZ_test_att0).squeeze().detach().cpu().numpy()
    df_test_PP= pd.DataFrame()
    df_test_PP_base= pd.DataFrame()

    [t1_pp, t0_pp] = threshold_pp(eta1_val, eta0_val, Y_val_att1_np, Y_val_att0_np)


    acc_pp=(((eta1_test >= t1_pp)==Y_test_att1_np).sum()+((eta0_test >= t0_pp)==Y_test_att0_np).sum())/test_size
    if (eta1_test >= t1_pp).mean() == 0:
        ppv1_fair = 1
    else:
        ppv1_fair=np.mean(Y_test_att1_np[eta1_test>= t1_pp])
    if (eta0_test >= t0_pp).mean() == 0:
        ppv0_fair = 1
    else:
        ppv0_fair=np.mean(Y_test_att0_np[eta0_test >= t0_pp])


    yyhateq11 = (Y_test_att1_np[eta1_test>= t1_pp]).sum()  + (Y_test_att0_np[eta0_test >= t0_pp] ).sum()
    yhateq1 = (eta1_test>= t1_pp).sum()  +(eta0_test>= t0_pp).sum()
    ppv_fair_mean=yyhateq11/yhateq1
    # DPP= np.max([abs(ppv1_fair-ppv_fair_mean) ,abs(ppv0_fair-ppv_fair_mean)])
    DPP= abs(ppv1_fair-ppv0_fair)


    acc_pp_base = (((eta1_test >= 0.5)==Y_test_att1_np).sum()+((eta0_test >= 0.5)==Y_test_att0_np).sum())/test_size
    if (eta1_test > 0.5).mean() == 0:
        ppv1 = 1
    else:
        ppv1=np.mean(Y_test_att1_np[eta1_test >= 0.5])

    if (eta0_test > 0.5).mean() == 0:
        ppv0 = 1
    else:
        ppv0=np.mean(Y_test_att0_np[eta0_test >= 0.5])
    yyhateq11_base = np.sum(Y_test_att1_np[eta1_test >= 0.5]) + np.sum(Y_test_att0_np[eta0_test >= 0.5])
    yhateq1_base = np.sum(eta1_test >= 0.5) + np.sum(eta0_test >= 0.5)
    ppv_fair_mean_base = yyhateq11_base / yhateq1_base
    # DPP_base = np.max([abs(ppv1 - ppv_fair_mean_base), abs(ppv0 - ppv_fair_mean_base)])
    DPP_base = abs(ppv1-ppv0)
    data = [t1_pp,t0_pp,acc_pp,DPP,dataset_name,seed]
    columns = ['PP_t1','PP_t0','acc_PP','DPP','dataset_name','seed']
    df_test_temp=pd.DataFrame([data], columns=columns)
    data_base = [acc_pp_base,DPP_base,dataset_name,seed]
    columns = ['acc_PP','DPP','dataset_name','seed']
    df_test_temp_base=pd.DataFrame([data_base], columns=columns)
    df_test_PP=df_test_PP.append(df_test_temp)
    df_test_PP_base=df_test_PP_base.append(df_test_temp_base)


    return [df_test_PP,df_test_PP_base]

