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
from .train import W_Train, C_Train_dp, C_Train_nofair, V_Train



####ADDED BY GRACE K 10/17/2022####
def get_labels(data, protected):
    if data == 'compas': 
        if protected == 'gender':
            priv='Male'
            #fair_col='Sex_Code_Text'
            fair_col='sex'
        if protected == 'race':
            priv='Caucasian'
            #fair_col='Ethnic_Code_Text'
            fair_col = 'race'
        #y_col='binary_text'
        y_col='two_year_recid'
        #pos='Low_Chance'
        pos='0'
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
        pos = '1'
    if data == 'bank':
        if protected=='marital':
            fair_col='marital'
            priv='married'
        if protected=='age':
            fair_col='age'
            priv='1'
        y_col='y'
        pos='yes'
    if data == 'medical':
        if protected=='race':
            fair_col='RACE'
            priv='1'
        y_col='UTILIZATION'
        pos='1'
    return priv, pos, fair_col, y_col


def C_Train(t, path, sampleloader, G, D, Enc, epochs, lr, dataloader, z_dim, dataset, col_type, sample_times, data, protected, itertimes = 100, steps_per_epoch = None, GPU=False):
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
        Enc.cuda()
        G.GPU = True
    
    all_labels = dataloader.label
    conditions = np.unique(all_labels.view(all_labels.dtype.descr * all_labels.shape[1]))
    D_optim = optim.Adam(D.parameters(), lr=lr, weight_decay=0.0001)
    G_optim = optim.Adam(G.parameters(), lr=lr, weight_decay=0.0001)
    ##FD (Fairness Discriminator) ADDED BY GRACE K 3/13/2023##
    priv, pos, fair_col, y_col = get_labels(data, protected)
    FD = VGAN_discriminator(dataset.dim-dataset.col_dim[dataset.columns.index(y_col)], 100, 1, False, 0)
    FD_optim = optim.Adam(FD.parameters(), lr=lr, weight_decay=0.0001)
    Enc_optim = optim.Adam(Enc.parameters(), lr=lr, weight_decay=0.0001)

    if GPU:
        FD.cuda()
    ##END ADDED 3/13/2023##

    # the default # of steps is the # of batches.
    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)

    eps = .000000000001
    eps = 0.0
 
    ###################
    #Only One Round
    ###################
    ae_count = 2 
    lam =2 
    for epoch in range(10):
        it = 0
        if it%100 == 0:
            log = open(path+"train_log_"+str(t)+".txt","a+")
            log.write("-----------Epoch {}-----------\n".format(epoch))
            log.close()
            print("-----------Epoch {}-----------".format(epoch))
	
        ##############################
        #Autoencoder
        ##############################
        while it < steps_per_epoch and epoch < ae_count:
            c=random.choice(conditions)
            x_real,c_real=dataloader.sample(label=list(c))

            if GPU:
                x_real=x_real.cuda()
                c_real=c_real.cuda()

            ##############################
            #Sample a batch of m examples Pdata
            ##############################
            m = x_real.shape[0]
            z_dim = x_real.shape[1]

            ''' train Autoencoder '''
            z = torch.randn(m, z_dim)
            if GPU:
                z = z.cuda()
            
            ##############################
            #Sample a batch of m examples PG
            ##############################
            e = Enc(x_real, c_real)
            x_fake = G(e, c_real)
            G_Loss = 1/m*torch.sum((x_fake-x_real)**2)
            
            ''' 
            x_fake_Enc = x_fake.clone().detach().requires_grad_(True)
            x_real_Enc = x_real.clone().detach().requires_grad_(True)
            Enc_Loss = -1/m*torch.sum((x_fake_Enc-x_real_Enc)**2)
            ''' 
            
            G_optim.zero_grad()
            #Enc_optim.zero_grad()
            G_Loss.backward() #Enc
            G_optim.step()
            #Enc_Loss.backward()
            #Enc_optim.step()

            it += 1
            
            if it%itertimes == 0:
                log = open(path+"train_log_"+str(t)+".txt","a+")
                ''' 
                log.write("iterator {}, G_Loss:{}, Enc_Loss:{}\n".format(it, G_Loss.data, Enc_Loss.data))
                print("iterator {}, G_Loss:{}, Enc_Loss:{}\n".format(it, G_Loss.data, Enc_Loss.data))
                ''' 
                log.write("iterator {}, G_Loss:{} \n".format(it, G_Loss.data))
                print("iterator {}, G_Loss:{}\n".format(it, G_Loss.data))
                log.close()

        ##############################
        #For Number of training iterations do
        ##############################
        print('Steps: ', steps_per_epoch)

        while it < steps_per_epoch and epoch >= ae_count:
            c=random.choice(conditions)
            x_real,c_real=dataloader.sample(label=list(c))
            if GPU:
                x_real=x_real.cuda()
                c_real=c_real.cuda()

            ##############################
            #Sample a batch of m examples Pdata
            ##############################

            m = x_real.shape[0]

            z = torch.randn(m, z_dim)
            if GPU:
                z = z.cuda()

            ##############################
            #Sample a batch of m examples PG
            ##############################
            
            e = Enc(z, c_real)
            x_fake = G(e, c_real)
            #x_fake = G(z, c_real)

            y_real = D(x_real, c_real)
            #print('real: ',torch.sum(y_real))
            y_fake = D(x_fake, c_real)
            #print('fake: ',torch.sum(y_fake))
            #print('fake: ',torch.sum(torch.sub(1,y_fake)))
            
            if GPU:
                y_real = y_real.cuda()
                y_fake = y_fake.cuda()
            
            ##############################
            # Update D1 by ascending its stochastic gradient
            ##############################
            y_fake_G = y_fake.clone().detach().requires_grad_(True)
            y_real_G = y_real.clone().detach().requires_grad_(True)
            D_Loss = -1/m*(torch.sum(torch.log(y_real+eps)+torch.log(torch.sub(1,y_fake)+eps)))
            
            #G_optim.zero_grad()
            D_optim.zero_grad()
            D_Loss.backward()
            D_optim.step()

            ##############################
            # Update G by descending its stochastic gradient
            ##############################

            z = torch.randn(m, z_dim)
            if GPU:
                z = z.cuda()
            
            e = Enc(z, c_real)
            x_fake = G(e, c_real)
            #x_fake = G(z, c_real)

            y_fake_G = D(x_fake, c_real)
            y_real_G = D(x_real, c_real)

            if GPU:
                y_fake_G = y_fake_G.cuda()
                y_real_G = y_real_G.cuda()
            
            G_Loss = 1/m*(torch.sum(torch.log(torch.sub(1,y_fake_G)+eps)))
            #G_Loss = 1/m*(torch.sum(torch.log(y_real_G+eps)+torch.log(torch.sub(1,y_fake_G)+eps)))
            DG_Loss = G_Loss.data

            G_optim.zero_grad()
            #D_optim.zero_grad()
            #FD_optim.zero_grad()
            G_Loss.backward() #D
            G_optim.step()

            #DG_Loss = torch.tensor(0.0)
            #D_Loss = torch.tensor(0.0)

            ##############################
            # D2
            ##############################
            #fair_idx = dataset.columns.index(fair_col)
            #nonfair_idx = np.delete(np.arange(x_fake.size()[1]), fair_idx)

            ##############################
            # Sample a batch of m examples s=1 and a batch of m examples s=0 
            ##############################
            #train Discriminator 
            z = torch.randn(m, z_dim)
            ##c_s0, c_s1 in size of c_real
            c_s0 = torch.tensor([[0.,1.]]).repeat(c_real.shape[0],1)
            c_s1 = torch.tensor([[1.,0.]]).repeat(c_real.shape[0],1)
            
            if GPU:
                z = z.cuda()
                c_s0 = c_s0.cuda()
                c_s1 = c_s1.cuda()

            e = Enc(z, c_s0)
            x_fake_s0 = G(e, c_s0)
            #x_fake_s0 = G(z, c_s0)

            z = torch.randn(m, z_dim)
            if GPU:
                z = z.cuda()

            e = Enc(z, c_s1)
            x_fake_s1 = G(e, c_s1)
            #x_fake_s1 = G(z, c_s1)

            y_fake_s0 = FD(x_fake_s0)
            y_fake_s1 = FD(x_fake_s1)

            y_fake_s0_FD = y_fake_s0#.clone().detach().requires_grad_(True)
            y_fake_s1_FD = y_fake_s1#.clone().detach().requires_grad_(True)

            if GPU:
                y_fake_s0_FD = y_fake_s0_FD.cuda()
                y_fake_s1_FD = y_fake_s1_FD.cuda()
            
            ##############################
            # Update D2 by ascending its stochastic gradient 
            ##############################
            FD_Loss = -lam*1/(2*m)*torch.sum(torch.log(y_fake_s1_FD+eps)+torch.log(torch.sub(1,y_fake_s0_FD)+eps))
            
            #G_optim.zero_grad()
            FD_optim.zero_grad()
            FD_Loss.backward()
            FD_optim.step()

            ##############################
            # Update G by descending its stochastic gradient 
            ##############################
            #train Discriminator 
            z = torch.randn(m, z_dim)
            ##c_s0, c_s1 in size of c_real
            c_s0 = torch.tensor([[0.,1.]]).repeat(c_real.shape[0],1)
            c_s1 = torch.tensor([[1.,0.]]).repeat(c_real.shape[0],1)
            
            if GPU:
                z = z.cuda()
                c_s0 = c_s0.cuda()
                c_s1 = c_s1.cuda()

            e = Enc(z, c_s0)
            x_fake_s0 = G(e, c_s0)
            #x_fake_s0 = G(z, c_s0)

            z = torch.randn(m, z_dim)
            if GPU:
                z = z.cuda()

            e = Enc(z, c_s1)
            x_fake_s1 = G(e, c_s1)
            #x_fake_s1 = G(z, c_s1)

            y_fake_s0 = FD(x_fake_s0)
            y_fake_s1 = FD(x_fake_s1)

            y_fake_s0_G = y_fake_s0#.clone().detach().requires_grad_(True)
            y_fake_s1_G = y_fake_s1#.clone().detach().requires_grad_(True)

            if GPU:
                y_fake_s0_G = y_fake_s0_G.cuda()
                y_fake_s1_G = y_fake_s1_G.cuda()

            G_Loss = lam*1/(2*m)*torch.sum(torch.log(y_fake_s1_G+eps)+torch.log(torch.sub(1,y_fake_s0_G)+eps))

            G_optim.zero_grad()
            D_optim.zero_grad()
            FD_optim.zero_grad()
            G_Loss.backward() #FD
            G_optim.step()

            #FD_Loss = torch.tensor(0.0)
            
            it += 1

            if it%itertimes == 0:
                log = open(path+"train_log_"+str(t)+".txt","a+")
                log.write("iterator {}, FD_Loss:{}, G_Loss:{}, D_Loss:{}, DG_Loss:{}\n".format(it,FD_Loss.data, G_Loss.data, D_Loss.data, DG_Loss))
                print("iterator {}, FD_Loss:{}, G_Loss:{}, D_Loss:{}, DG_Loss:{}\n".format(it,FD_Loss.data, G_Loss.data, D_Loss.data, DG_Loss))
                log.close()
        G.eval()

        fair_idx = dataset.columns.index(fair_col)
        fair_idx = dataset.__dict__[fair_col].dim
        before = 0
        for col in dataset.columns:
           if col != fair_col:
               before = before+dataset.__dict__[col].dim 
           else:
               fair_idx = before 
        for time in range(sample_times):
            sample_data = None
            cond=random.choice(conditions)
            print('COLUMNS: ',dataset.columns)
            print('FAIR IDX: ',fair_idx)
            for x, y in sampleloader: #y = y
                z = torch.randn(x.shape[0], z_dim)
                if GPU:
                    z = z.cuda()
                    y = y.cuda()

                e = Enc(z, y)
                x_fake = G(e, y)
                #print('x_fake: ', x_fake.size())
                x_fake_0, x_fake_1 = torch.split(x_fake, [fair_idx,x_fake.size()[1]-fair_idx],dim=1)
                x_fake = torch.cat((x_fake_0, y), dim = 1)
                x_fake = torch.cat((x_fake, x_fake_1), dim = 1)
                #print('x_fake: ', x_fake.size())
                #x_fake = torch.cat((y,x_fake), dim = 1)
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
            sample_data.to_csv(path+'sample_data_FAIRGAN_'+protected+'_{}_{}_{}.csv'.format(t,epoch,time), index = None)
        G.train()        
    return G,D
