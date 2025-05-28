import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
import math

from synthesizer import VGAN_discriminator

####ADDED BY GRACE K 10/17/2022####
#def FLoss(x_fake, dataset, data, protected):
def get_labels(data, protected):
    if data == 'compas': 
        if protected == 'gender':
            priv='Male'
            fair_col='Sex_Code_Text'
        if protected == 'race':
            priv='Caucasian'
            fair_col='Ethnic_Code_Text'
        y_col='binary_text'
        pos='Low_Chance'
    if data == 'adult':
        if protected == 'gender':
            priv='Male'
            fair_col='sex'
        if protected == 'race':
            priv='White'
            fair_col='race'
        y_col='income'
        pos='>50K'
    if data == 'census':
        if protected == 'gender':
            priv=' Male'
            fair_col='SEX'
        if protected == 'race':
            priv=' White'
            fair_col='RACE'
        y_col='INCOME_50K'
        pos=' 50000+.'
    if data == 'german':
        if protected == 'race':
            print("NO RACE IN THIS DATASET")
            return
        if protected == 'gender':
            priv='male'
            fair_col='gender'
        y_col='labels'
        pos='1'
    if data == 'bank':
        if protected == 'age':
            priv='1'
            fair_col='age'
        y_col='y'
        pos='yes'
    return priv, pos, fair_col, y_col

def FLoss(x_fake, dataset, priv, pos, fair_col, y_col):
    x=dataset.reverse(x_fake.cpu().detach().numpy())
    if fair_col in dataset.columns:
        fair_idx=dataset.columns.index(fair_col) 
        x[:,fair_idx] = np.where(x[:,fair_idx].astype(str) == priv, 1, 0)
        fair=x[:,fair_idx]
    else:
        print("X NOT FOUND")
    if y_col in dataset.columns:
        y_idx=dataset.columns.index(y_col)
        x[:,y_idx] = np.where(x[:,y_idx].astype(str) == pos, 1, 0)
        y=x[:,y_idx] 
    else:
        print("Y_COL NOT FOUND")
    fair_y=np.stack((fair,y),axis=1)
    counts=pd.DataFrame(fair_y).value_counts().unstack(fill_value=0).stack()#.rename(index={priv:1,'Unprivelaged':0})#,columns={pos:1,'Undesired':0})
    #print(counts)
    try:
        num=(counts[0][1]/sum(counts[0]))
    except:
        num=1
    try:
        denom=(counts[1][1]/sum(counts[1]))
    except:
        denom=-1
    #disp=abs(num-denom)
    disp=1-min(num/denom, 1-num/denom)
    return disp
####END ADDED####

# compute kl loss (not use now)    
def compute_kl(real, pred):
    return torch.sum((torch.log(pred + 1e-4) - torch.log(real + 1e-4)) * pred)

def KL_Loss(x_fake, x_real, col_type, col_dim, dataset, y_idx):
    ###ADDED 1/3/2023###
    # Get Predicted

    kl = 0.0
    sta = 0
    end = 0
    for i in range(len(col_type)):
        if i != y_idx:
            dim = col_dim[i]
            sta = end
            end = sta+dim
            fakex = x_fake[:,sta:end]
            realx = x_real[:,sta:end]
            if col_type[i] == "gmm":
                fake2 = fakex[:,1:]
                real2 = realx[:,1:]
                dist = torch.sum(fake2, dim=0)
                dist = dist / torch.sum(dist)
                real = torch.sum(real2, dim=0)
                real = real / torch.sum(real)
                kl += compute_kl(real, dist)
            else:
                dist = torch.sum(fakex, dim=0)
                dist = dist / torch.sum(dist)
                
                real = torch.sum(realx, dim=0)
                real = real / torch.sum(real)
                
                kl += compute_kl(real, dist)
    return kl
    
##ADDED FROM TABFAIRGAN BY GRACE K 3/1/2023##
def get_gradient(crit, real, fake, epsilon):
    mixed_data = real*epsilon+fake*(1-epsilon)
    mixed_scores = crit(mixed_data)
    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient
def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient, -1))
    gradient_norm = gradient.norm(2, dim=1)
    penalty=torch.mean((gradient_norm - 1) ** 2)
    return penalty
##END ADDED 3/1/2023##

def V_Train(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times, data, protected, itertimes = 100, steps_per_epoch = None, GPU=False, KL=True):
    """
    The vanilla (basic) training process for GAN
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sample_rows: # of synthesized rows
        * G: the generator
        * D: the discriminator
        * epochs: # of epochs
        * lr: learning rate
        * dataloader: the data loader
        * z_dim: dimension of noise
        * dataset: the dataset for reversible data tranformation
        * itertimes:
        * steps_per_epoch: # of steps per epoch
        * GPU: the GPU flag
        ####ADDED BY GRACE K 10/18/2022####
        * data: dataset name
        * protected: fairness column 'gender' or 'race'
        ####END ADDED 10/18/2022####
    Return:
        * G: the generator
        * D: the descriminator
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)
    ##FD (Fairness Discriminator) ADDED BY GRACE K 3/13/2023##
    priv, pos, fair_col, y_col = get_labels(data, protected)
    FD = VGAN_discriminator(dataset.dim-dataset.col_dim[dataset.columns.index(y_col)]+1, 100, 1, False, 0)#(x_dim, param["dis_hidden_dim"], param["dis_num_layers"], condition, c_dim)
    FD_optim = optim.Adam(FD.parameters(), lr=lr, weight_decay=0.00001)
    if GPU:
        FD.cuda()
    ##END ADDED 3/13/2023##
    ## ADDED BY GRACE K 3/28/2023##
    both = False
    if protected == 'both':
        both = True
    if both == True:
        protected='race'
        priv, pos, fair_col, y_col = get_labels(data, protected)
        FD_1 = VGAN_discriminator(dataset.dim-dataset.col_dim[dataset.columns.index(y_col)]+1, 100, 1, False, 0)#(x_dim, param["dis_hidden_dim"], param["dis_num_layers"], condition, c_dim)
        FD_1_optim = optim.Adam(FD_1.parameters(), lr=lr, weight_decay=0.00001)
        if GPU:
            FD_1.cuda()
    ##END ADDED 3/28/2023

    # the default # of steps is the # of batches.
    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

    for epoch in range(epochs):
        it = 0
        log = open(path+"train_log_"+str(t)+".txt","a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
	

        priv, pos, fair_col, y_col = get_labels(data, protected)

        while it < steps_per_epoch:
            for x_real in dataloader:
                if GPU:
                    x_real = x_real.cuda()
                print(x_real.shape)

                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                
                x_fake = G(z)

                if epoch > -1: #%2 == 0 or epoch < 5: #Use real every other epoch
                    fair_disc = False
                    y_real = D(x_real)
                    y_fake = D(x_fake)
                
                    # D_Loss = -(torch.mean(y_real) - torch.mean(y_fake)) # wgan loss
                    fake_label = torch.zeros(y_fake.shape[0], 1)
                    real_label = np.ones([y_real.shape[0], 1])
                    # Avoid the suppress of Discriminator over Generator
                    real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                    real_label = torch.from_numpy(real_label).float()
                    if GPU:
                        fake_label = fake_label.cuda()
                        real_label = real_label.cuda()
                    
                    D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                    D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                    D_Loss = D_Loss1 + D_Loss2
                    ###TEST FROM TABFAIRGAN ADDED BY GRACE K 3/13/2023##
                    #gradient = get_gradient(D, x_real.float(), x_fake.detach(), z)
                    #gp = gradient_penalty(gradient)
                    #D_Loss = D_Loss + 10*gp 
                    ### END ADDED 3/13/2023 ###
                
                    G_optim.zero_grad()
                    D_optim.zero_grad()
                    D_Loss.backward()
                    D_optim.step()
                    ##FOR ONE DISC LOSS ADDED BY GRACE K 3/26/23
                    if epoch > 5:
                        protected='gender'
                        priv, pos, fair_col, y_col = get_labels(data, protected)
                        if GPU:
                            x_real = x_real.cuda()

                        ''' train Discriminator '''
                        z = torch.randn(x_real.shape[0], z_dim)
                        if GPU:
                            z = z.cuda()
                        
                        x_fake = G(z)
                        fair_disc = True
                        fair_idx = dataset.columns.index(fair_col)
                        nonfair_idx = np.delete(np.arange(x_fake.size()[1]), fair_idx)

                        x_fake_noProt = x_fake[:,nonfair_idx]
                        x_real_noProt = x_real[:,nonfair_idx]
                        fair_real_label = x_fake[:,[fair_idx]].float()
                        fair_fake_label = FD(x_fake_noProt)

                        if GPU:
                            fair_real_label = fair_real_label.cuda()
                            fair_fake_label = fair_fake_label.cuda()
                    
                        FD_Loss = F.binary_cross_entropy(fair_fake_label, fair_real_label.detach())
                        G_optim.zero_grad()
                        FD_optim.zero_grad()
                        FD_Loss.backward()
                        FD_optim.step()

                    ##END ADDED 3/26/23 ##########################
                    ##FOR TWO PROTECTED VALUES ADDED BY GRACE K 3/28/23
                    if epoch > 5 and both == True:
                        protected='race'
                        priv, pos, fair_col, y_col = get_labels(data, protected)
                        if GPU:
                            x_real = x_real.cuda()

                        ''' train Discriminator '''
                        z = torch.randn(x_real.shape[0], z_dim)
                        if GPU:
                            z = z.cuda()
                        
                        x_fake = G(z)
                        fair_idx = dataset.columns.index(fair_col)
                        nonfair_idx = np.delete(np.arange(x_fake.size()[1]), fair_idx)

                        x_fake_noProt = x_fake[:,nonfair_idx]
                        x_real_noProt = x_real[:,nonfair_idx]
                        fair_real_label = x_fake[:,[fair_idx]].float()
                        fair_fake_label = FD_1(x_fake_noProt)

                        if GPU:
                            fair_real_label = fair_real_label.cuda()
                            fair_fake_label = fair_fake_label.cuda()
                    
                        FD_1_Loss = F.binary_cross_entropy(fair_fake_label, fair_real_label.detach())
                        G_optim.zero_grad()
                        FD_1_optim.zero_grad()
                        FD_1_Loss.backward()
                        FD_1_optim.step()
                    ##END ADDED 3.28.23##



                    ''' train Generator '''
                    z = torch.randn(x_real.shape[0], z_dim)
                    if GPU:
                        z = z.cuda()
                
                    x_fake = G(z)
                    y_fake = D(x_fake)
                
                    real_label = torch.ones(y_fake.shape[0], 1)
                    if GPU:
                        real_label = real_label.cuda()
                    G_Loss1 = F.binary_cross_entropy(y_fake, real_label)

                    if KL:
                        KL_loss =  KL_Loss(x_fake, x_real, col_type, dataset.col_dim, dataset, dataset.columns.index(y_col))
                        G_Loss = G_Loss1 + KL_loss
                    else:
                        G_Loss = G_Loss1
                    ####ADDED BY GRACE K 3/26/23
                    if fair_disc == True:
                        protected='gender'
                        priv, pos, fair_col, y_col = get_labels(data, protected)
                        x_fake_noProt = x_fake[:,nonfair_idx]
                        fair_real_label = x_fake[:,[fair_idx]] #GET PRIVILEGED INDEX OF given data
                        fair_fake_label = FD(x_fake_noProt)

                        if GPU:
                             fair_real_label = fair_real_label.cuda()
                        G_Loss_fair = -F.binary_cross_entropy(fair_fake_label, fair_real_label.detach())
                        G_Loss = G_Loss + G_Loss_fair
                    ####END ADDED 3/26/23
                    if fair_disc == True and both == True:
                        protected='race'
                        priv, pos, fair_col, y_col = get_labels(data, protected)
                        x_fake_noProt = x_fake[:,nonfair_idx]
                        fair_real_label = x_fake[:,[fair_idx]] #GET PRIVILEGED INDEX OF given data
                        fair_fake_label = FD_1(x_fake_noProt)

                        if GPU:
                             fair_real_label = fair_real_label.cuda()
                        G_Loss_fair = -F.binary_cross_entropy(fair_fake_label, fair_real_label.detach())
                        G_Loss = G_Loss + G_Loss_fair
                    ####END ADDED 3/26/23

                    ####ADDED BY GRACE K 10/17/2022####
                    #if epoch > epochs/2: 
                    #    test_fairness = True
                    #else:
                    #    test_fairness = False 
                    test_fairness = True
                    if test_fairness:
                        protected='gender'
                        priv, pos, fair_col, y_col = get_labels(data, protected)
                        G_Loss = G_Loss + FLoss(x_fake, dataset, priv, pos, fair_col, y_col)
                    if test_fairness and both == True:
                        protected='race'
                        priv, pos, fair_col, y_col = get_labels(data, protected)
                        G_Loss = G_Loss + FLoss(x_fake, dataset, priv, pos, fair_col, y_col)
                     
                    ####END ADDED####
                    G_optim.zero_grad()
                    D_optim.zero_grad()
                    if fair_disc == True:
                        FD_optim.zero_grad()
                    if fair_disc == True and both == True:
                        FD_1_optim.zero_grad()
                    G_Loss.backward()
                    G_optim.step()

                else:
                    print("IN FAIR DISCRIMINATOR")
                    fair_disc = True
                    priv, pos, fair_col, y_col = get_labels(data, protected)
                    fair_idx = dataset.columns.index(fair_col)
                    nonfair_idx = np.delete(np.arange(x_fake.size()[1]), fair_idx)

                    x_fake_noProt = x_fake[:,nonfair_idx]
                    x_real_noProt = x_real[:,nonfair_idx]
                    real_label = x_fake[:,[fair_idx]].float()
                    fake_label = FD(x_fake_noProt)

                    if GPU:
                        real_label = real_label.cuda()
                        fake_label = fake_label.cuda()

                    
                    FD_Loss = F.binary_cross_entropy(fake_label, real_label.detach())
                    ###TEST FROM TABFAIRGAN ADDED BY GRACE K 3/13/2023##
                    #gradient = get_gradient(FD, x_real_noProt.float(), x_fake_noProt.detach(), z[x_real_noProt.size()[0], x_real_noProt.size()[1]])
                    #gp = gradient_penalty(gradient)
                    #FD_Loss = FD_Loss + 10*gp 
                    ### END ADDED 3/13/2023 ###
                
                    G_optim.zero_grad()
                    FD_optim.zero_grad()
                    FD_Loss.backward()
                    FD_optim.step()

                    ''' train Generator '''
                    z = torch.randn(x_real.shape[0], z_dim)
                    if GPU:
                        z = z.cuda()
                
                    x_fake = G(z)
                
                    x_fake_noProt = x_fake[:,nonfair_idx]
                    x_real_noProt = x_real[:,nonfair_idx]
                    real_label = x_fake[:,[fair_idx]] #GET PRIVILEGED INDEX OF given data
                    fake_label = FD(x_fake_noProt)

                    if GPU:
                        real_label = real_label.cuda()
                    G_Loss1 = -F.binary_cross_entropy(fake_label, real_label.detach())

                    if KL:
                        KL_loss =  KL_Loss(x_fake, x_real, col_type, dataset.col_dim, dataset, dataset.columns.index(y_col))
                        G_Loss = G_Loss1 + KL_loss
                    else:
                        G_Loss = G_Loss1

                    G_optim.zero_grad()
                    FD_optim.zero_grad()
                    G_Loss.backward()
                    G_optim.step()


                it += 1

                if it%itertimes == 0:
                    log = open(path+"train_log_"+str(t)+".txt","a+")
                    if fair_disc == True:
                        log.write("iterator {}, FD_Loss:{}, G_Loss:{}\n".format(it,FD_Loss.data, G_Loss.data))
                        #log.close()
                        print("iterator {}, FD_Loss:{}, G_Loss:{}\n".format(it,FD_Loss.data, G_Loss.data))
                    #else:
                    #if fair_disc == True:
                    log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                    log.close()
                    print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                if it >= steps_per_epoch:
                    G.eval()
                    #if GPU:
                    #    G.cpu()
                    #    G.GPU = False
                    for time in range(sample_times):
                        sample_data = None
                        for x_real in sampleloader:
                            z = torch.randn(x_real.shape[0], z_dim)
                            #print('x_real: ',str(x_real.shape[0]))
                            #print('z_dim: ', str(z_dim))
                            #print('z size: '+str(z.shape))
                            if GPU:
                                z = z.cuda()
                            x_fake = G(z)
                            samples = x_fake
                            samples = samples.reshape(samples.shape[0], -1)
                            samples = samples[:,:dataset.dim]
                            samples = samples.cpu()
                            sample_table = dataset.reverse(samples.detach().numpy())
                            df = pd.DataFrame(sample_table,columns=dataset.columns)
                            #print(df)
                            if sample_data is None:
                                sample_data = df
                            else:
                                sample_data = sample_data.append(df)
                        sample_data.to_csv(path+'sample_data_{}_{}_{}.csv'.format(t,epoch,time), index = None)
                   # if GPU:
                   #     G.cuda()
                    #    G.GPU = True
                    G.train()
                    break
    return G,D


def W_Train(t, path, sampleloader, G, D, ng, nd, cp, lr, dataloader, z_dim, dataset, col_type, sample_times, itertimes = 100, GPU=False, KL=True):
    """
    The WGAN training process for GAN
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sample_rows: # of synthesized rows
        * G: the generator
        * D: the discriminator
        * ng: 
        * nd:
        * cp:
        * lr: learning rate
        * dataloader: the data loader
        * z_dim: dimension of noise
        * dataset: the dataset for reversible data tranformation
        * itertimes:
        * steps_per_epoch: # of steps per epoch
        * GPU: the GPU flag
    Return:
        * G: the generator
        * D: the descriminator
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    
    D_optim = optim.RMSprop(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.RMSprop(G.parameters(), lr=lr, weight_decay=0.00001)
        
    epoch_time = int(ng/100)
    # the default # of steps is the # of batches.

    for t1 in range(ng):
        for t2 in range(nd):
            x_real = dataloader.sample(dataloader.batch_size)
            if GPU:
                x_real = x_real.cuda()

            ''' train Discriminator '''
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()
            
            x_fake = G(z)

            y_real = D(x_real)
            y_fake = D(x_fake)
                
            D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))
            
            G_optim.zero_grad()
            D_optim.zero_grad()
            D_Loss.backward()
            D_optim.step()
            for p in D.parameters():
                p.data.clamp_(-cp, cp)  # clip the discriminator parameters (wgan)

        ''' train Generator '''
        z = torch.randn(dataloader.batch_size, z_dim)
        if GPU:
            z = z.cuda()
        x_fake = G(z)
        y_fake = D(x_fake)
        G_Loss1 = -torch.mean(y_fake)
        if KL:
            KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
            G_Loss = G_Loss1 + KL_loss
        else:
            G_Loss = G_Loss1
        G_optim.zero_grad()
        D_optim.zero_grad()
        G_Loss.backward()
        G_optim.step()

        if t1 % itertimes == 0:    
            print("------ng : {}-------".format(t1))
            print("generator loss: {}".format(G_Loss.data))
            print("discriminator loss: {}".format(D_Loss.data))
            log = open(path+"train_log_"+str(t)+".txt","a+")
            log.write("----------ng: {}---------\n".format(t1))
            log.write("generator loss: {}\n".format(G_Loss.data))
            log.write("discriminator loss: {}\n".format(D_Loss.data))
            log.close()  
        if t1 % epoch_time == 0 and t1 > 0:
            G.eval()
           # if GPU:
            #    G.cpu()
            #    G.GPU = False
            for time in range(sample_times):
                sample_data = None
                for x_real in sampleloader:
                    z = torch.randn(x_real.shape[0], z_dim)
                    if GPU:
                        z = z.cuda()
                    x_fake = G(z)
                    samples = x_fake
                    samples = samples.reshape(samples.shape[0], -1)
                    samples = samples[:,:dataset.dim]
                    samples = samples.cpu()
                    sample_table = dataset.reverse(samples.detach().numpy())
                    df = pd.DataFrame(sample_table,columns=dataset.columns)
                    if sample_data is None:
                        sample_data = df
                    else:
                        sample_data = sample_data.append(df)
                sample_data.to_csv(path+'sample_data_{}_{}_{}.csv'.format(t,int(t1/epoch_time),time), index = None)
            #if GPU:
            #    G.cuda()
            #    G.GPU = True
            G.train()
    return G,D


def C_Train(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times,data,protected, itertimes = 100, steps_per_epoch = None, GPU=False):
    """
    The
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sampleloader:
    :param G:
    :param D:
    :param epochs:
    :param lr:
    :param dataloader:
    :param z_dim:
    :param dataset:
    :param itertimes:
    :param steps_per_epoch:
    :param GPU:
    ####ADDED BY GRACE K 10/18/2022####
    :param data
    :param protected
    ####END ADDED 10/18/2022####
    :return:
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    print("CUDA DEVICE COUNT")
    print(torch.cuda.device_count())
    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)    
    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)
    for epoch in range(epochs):
        if epoch > -1:
            cond = True 
        else:
            cond = False
        log = open(path+"train_log_"+str(t)+".txt","a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.write(str(cond))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
        for it in range(steps_per_epoch):
	    ###EDITED BY GRACE K 10_5_2022 - TESTING SPLIT CONDITIONAL TRAINING FOR CLASS DISTRIBUTION###
            if cond:
                c = random.choice(conditions)
                x_real, c_real = dataloader.sample(label=list(c))
                if GPU:
                    x_real = x_real.cuda()
                    c_real = c_real.cuda()
            else:
                for x_real, c_real in dataloader:            
                    if GPU:
                        x_real = x_real.cuda()
                        c_real = c_real.cuda()
	    ###END EDITED BY GRACE K 10_5_2022####
            ''' train Discriminator '''
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()
                
            x_fake = G(z, c_real)
            y_real = D(x_real, c_real)
            y_fake = D(x_fake, c_real)
            
            #D_Loss = -(torch.mean(y_real) - torch.mean(y_fake)) # wgan loss
            fake_label = torch.zeros(y_fake.shape[0], 1)
            real_label = np.ones([y_real.shape[0], 1])
            real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
            real_label = torch.from_numpy(real_label).float()
            if GPU:
                fake_label = fake_label.cuda()
                real_label = real_label.cuda()
            
            D_Loss1 = F.binary_cross_entropy(y_real, real_label)
            D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
            D_Loss = D_Loss1 + D_Loss2
            
            G_optim.zero_grad()
            D_optim.zero_grad()
            D_Loss.backward()
            D_optim.step()
            ''' train Generator '''
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()
                
            x_fake = G(z, c_real)
            y_fake = D(x_fake, c_real)
            
            real_label = torch.ones(y_fake.shape[0], 1)
            if GPU:
                real_label = real_label.cuda()
                
            G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
            KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
            G_Loss = G_Loss1 + KL_loss
            ####ADDED BY GRACE K 10/17/2022####
            test_fairness = True
            if test_fairness:
                G_Loss = G_Loss + FLoss(torch.cat((x_fake,c_real),dim=1), dataset,data,protected)
            ####END ADDED 10/17/2022####

            G_optim.zero_grad()
            D_optim.zero_grad()
            G_Loss.backward()
            G_optim.step()

            if it%itertimes == 0:
                log = open(path+"train_log_"+str(t)+".txt","a+")
                log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                log.close()
                print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))

        G.eval()
        #if GPU:
        #    G.cpu()
        #    G.GPU = False
        for time in range(sample_times):
            sample_data = None
            for x, y in sampleloader:
                z = torch.randn(x.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                    y = y.cuda()
                x_fake = G(z, y)
                x_fake = torch.cat((x_fake, y), dim = 1)
                samples = x_fake
                samples = samples.reshape(samples.shape[0], -1)
                samples = samples[:,:dataset.dim]
                samples = samples.cpu()
                sample_table = dataset.reverse(samples.detach().numpy())
                df = pd.DataFrame(sample_table,columns=dataset.columns)
                if sample_data is None:
                    sample_data = df
                else:
                    sample_data = sample_data.append(df)
            sample_data.to_csv(path+'sample_data_{}_{}_{}.csv'.format(t,epoch,time), index = None)
        #if GPU:
        #    G.cuda()
        #    G.GPU = True
        G.train()
    return G,D

def C_Train_nofair(t, path, sampleloader, G, D, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times, data, protected, itertimes = 100, steps_per_epoch = None, GPU=False):
    """
    The
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sampleloader:
    :param G:
    :param D:
    :param epochs:
    :param lr:
    :param dataloader:
    :param z_dim:
    :param dataset:
    :param itertimes:
    :param steps_per_epoch:
    :param GPU:
    ####ADDED BY GRACE K 10/18/2022####
    :param data
    :param protected
    ####END ADDED 10/18/2022####
    :return:
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    print("Conditions: ", conditions)
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.00001)
        
    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)
    for epoch in range(epochs):
        log = open(path+"train_log_"+str(t)+".txt","a+")
        log.write("-----------Epoch {}-----------\n".format(epoch))
        log.close()
        print("-----------Epoch {}-----------".format(epoch))
        it = 0
        while it < steps_per_epoch:
            for x_real, c_real in dataloader:
                if GPU:
                    x_real = x_real.cuda()
                    c_real = c_real.cuda()
                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()           
                x_fake = G(z, c_real)
                y_real = D(x_real, c_real)
                y_fake = D(x_fake, c_real)
                fake_label = torch.zeros(y_fake.shape[0], 1)
                real_label = np.ones([y_real.shape[0], 1])
                real_label = real_label * 0.7 + np.random.uniform(0, 0.3, real_label.shape)
                real_label = torch.from_numpy(real_label).float()
                if GPU:
                    fake_label = fake_label.cuda()
                    real_label = real_label.cuda()
                
                D_Loss1 = F.binary_cross_entropy(y_real, real_label)
                D_Loss2 = F.binary_cross_entropy(y_fake, fake_label)
                D_Loss = D_Loss1 + D_Loss2
                
                G_optim.zero_grad()
                D_optim.zero_grad()
                D_Loss.backward()
                D_optim.step()
                ''' train Generator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                    
                x_fake = G(z, c_real)
                y_fake = D(x_fake, c_real)
                
                real_label = torch.ones(y_fake.shape[0], 1)
                if GPU:
                    real_label = real_label.cuda()
                    
                G_Loss1 = F.binary_cross_entropy(y_fake, real_label)
                KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
                G_Loss = G_Loss1 + KL_loss
                ####ADDED BY GRACE K 10/17/2022####
                if epoch > epochs/2: 
                    test_fairness = True
                else:
                    test_fairness = False 
                #test_fairness = True
                if test_fairness:
                    G_Loss = G_Loss + FLoss(x_fake, dataset, data, protected)
                ####END ADDED####

                G_optim.zero_grad()
                D_optim.zero_grad()
                G_Loss.backward()
                G_optim.step()
                it += 1

                if it%itertimes == 0:
                    log = open(path+"train_log_"+str(t)+".txt","a+")
                    log.write("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                    log.close()
                    print("iterator {}, D_Loss:{}, G_Loss:{}\n".format(it,D_Loss.data, G_Loss.data))
                    
                if it >= steps_per_epoch:
                    G.eval()
                    #if GPU:
                    #    G.cpu()
                    #    G.GPU = False
                    for time in range(sample_times):
                        sample_data = None
                        for x, y in sampleloader:
                            z = torch.randn(x.shape[0], z_dim)
                            if GPU:
                                z = z.cuda()
                                y = y.cuda()
                            x_fake = G(z, y)
                            x_fake = torch.cat((x_fake, y), dim = 1)
                            samples = x_fake
                            samples = samples.reshape(samples.shape[0], -1)
                            samples = samples[:,:dataset.dim]
                            samples = samples.cpu()
                            sample_table = dataset.reverse(samples.detach().numpy())
                            df = pd.DataFrame(sample_table,columns=dataset.columns)
                            if sample_data is None:
                                sample_data = df
                            else:
                                sample_data = sample_data.append(df)
                        sample_data.to_csv(path+'sample_data_{}_{}_{}.csv'.format(t,epoch, time), index = None)
                    #if GPU:
                    #    G.cuda()
                    #    G.GPU = True
                    G.train()
                    break
    return G,D


def C_Train_dp(t, path, sampleloader,G, D, ng, nd, cp, lr, dataloader, z_dim, dataset, col_type, eps, sample_times, itertimes = 100, GPU=False,delta=0.00001):
    """
    The Conditional Training with Differential Privacy
    Args:
        * t: the t-th training
        * path: the path for storing the log
        * sampleloader: 
        * G: the generator
        * D: the discriminator
        * ng: 
        * nd:
        * cp:
        * lr: learning rate
        * dataloader: the data loader
        * z_dim: dimension of noise
        * dataset: the dataset for reversible data tranformation
        * itertimes:
        * steps_per_epoch: # of steps per epoch
        * GPU: the GPU flag
    Return:
        * G: the generator
        * D: the descriminator
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if GPU:
        G.cuda()
        D.cuda()
        G.GPU = True
    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.RMSprop(D.parameters(), lr=lr, weight_decay=0.00001)
    G_optim = optim.RMSprop(G.parameters(), lr=lr, weight_decay=0.00001)
    
    q = dataloader.batch_size / len(dataloader.data)
    theta_n = 2*q*math.sqrt(nd*math.log(1/delta)) / eps     
    epoch_time = int(ng/5)
    print("theta_n: {}".format(theta_n))
    # the default # of steps is the # of batches.

    for t1 in range(ng):
        for c in conditions:
            for t2 in range(nd):
                x_real, c_real = dataloader.sample(label=list(c))
                if GPU:
                    x_real = x_real.cuda()
                    c_real = c_real.cuda()
                ''' train Discriminator '''
                z = torch.randn(x_real.shape[0], z_dim)
                if GPU:
                    z = z.cuda()

                x_fake = G(z, c_real)
                      
                y_real = D(x_real, c_real)
                y_fake = D(x_fake, c_real)
                    
                D_Loss = -(torch.mean(y_real) - torch.mean(y_fake))
                
                D_optim.zero_grad()
                G_optim.zero_grad()
                D_Loss.backward()
                
                for p in D.parameters():
                    sigma = theta_n * 1
                    noise = np.random.normal(0, sigma, p.grad.shape) / dataloader.batch_size
                    noise = torch.from_numpy(noise).float()
                    if GPU:
                        noise = noise.cuda()
                    p.grad += noise
                
                D_optim.step()
                for p in D.parameters():
                    p.data.clamp_(-cp, cp)  # clip the discriminator parameters (wgan)

            ''' train Generator '''
            x_real, c_real = dataloader.sample(label=list(c))
            z = torch.randn(x_real.shape[0], z_dim)
            if GPU:
                z = z.cuda()
                c_real = c_real.cuda()
            x_fake = G(z, c_real)
            y_fake = D(x_fake, c_real)
            G_Loss = -torch.mean(y_fake)
           # KL_loss = KL_Loss(x_fake, x_real, col_type, dataset.col_dim)
            #G_Loss = G_Loss1 + KL_loss
            G_optim.zero_grad()
            D_optim.zero_grad()
            G_Loss.backward()
            G_optim.step()

        if t1 % itertimes == 0:    
            print("------ng : {}-------".format(t1))
            print("generator loss: {}".format(G_Loss.data))
            print("discriminator loss: {}".format(torch.mean(D_Loss).data))
            log = open(path+"train_log_"+str(t)+".txt","a+")
            log.write("----------ng: {}---------\n".format(t1))
            log.write("generator loss: {}\n".format(G_Loss.data))
            log.write("discriminator loss: {}\n".format(torch.mean(D_Loss).data))
            log.close()  
        
        if (t1+1) % epoch_time == 0 and t1 > 0:
            G.eval()
            if GPU:
                G.cpu()
                G.GPU = False
            for time in range(sample_times):
                y = torch.from_numpy(sampleloader.label).float()
                z = torch.randn(len(sampleloader.label), z_dim)
                x_fake = G(z, y)
                x_fake = torch.cat((x_fake, y), dim = 1)
                samples = x_fake.cpu()
                samples = samples.reshape(samples.shape[0], -1)
                samples = samples[:,:dataset.dim]
                sample_table = dataset.reverse(samples.detach().numpy())
                sample_data = pd.DataFrame(sample_table,columns=dataset.columns)
                sample_data.to_csv(path+'sample_data_{}_{}_{}_{}.csv'.format(eps, t,int(t1/epoch_time),time), index = None)
            if GPU:
                G.cuda()
                G.GPU = True
            G.train()
    return G,D
        
        
        
        
