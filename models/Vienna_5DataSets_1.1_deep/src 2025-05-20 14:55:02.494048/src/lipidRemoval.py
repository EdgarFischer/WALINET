import os
import sys
import numpy as np
import nibabel as nib
import h5py
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
import scipy
import torch
import torch.nn as nn




def runNNLipRemoval2(image_rrrf, headmask, skMask, device, exp):
    sta_epoch = time.time()

    #################################
    ### Lipid Projection Operator ###
    #################################

    s = image_rrrf.shape
    beta=1E+3 * .3 #0.938
    multBeta = 1.5
    lipidFac = 0
    lower=None

    Data_rf = np.reshape(image_rrrf, (s[0]*s[1]*s[2],s[3]))
    lipid_mask = np.reshape(skMask, (s[0]*s[1]*s[2]))

    lipid_rf = Data_rf[lipid_mask>0,:]

    while np.abs(lipidFac-0.938) > 0.005:
        LipidRem_Operator_ff = np.linalg.inv(np.eye(s[-1]) + beta * np.matmul(np.conj(lipid_rf).T, lipid_rf))
        ## Mean vlaue of diagonal should be 1 or above 0.9
        ## 
        print("Mean Absolute Value of Diagonal of Lipid Suppression Operator: ")
        lipidFac = np.mean(np.abs(np.diagonal(LipidRem_Operator_ff)))
        print(lipidFac)

        if lipidFac < 0.938:
            beta = beta/multBeta
            if lower==False:
                multBeta=0.5*multBeta
            lower=True
        else:
            beta = beta*multBeta
            if lower==True:
                multBeta=0.5*multBeta
            lower=False

    LipidRem_Operator_ff = np.linalg.inv(np.eye(s[-1]) + beta * np.matmul(np.conj(lipid_rf).T, lipid_rf))
    LipidProj_Operator_ff = np.eye(s[-1])-LipidRem_Operator_ff
    #Data_ConvLipidRemoved_rrrf = np.reshape(np.matmul(Data_rf, LipidRem_Operator_ff), s)
    #Data_LipidProj_rrrf = np.reshape(np.matmul(Data_rf, LipidProj_Operator_ff), s)
    
    

    ####################
    ### Prepare Data ###
    ####################

    lip = image_rrrf[headmask>0,:]
    lipProj = np.matmul(lip, LipidProj_Operator_ff)
    lip = torch.tensor(lip, dtype=torch.cfloat)
    lipProj = torch.tensor(lipProj, dtype=torch.cfloat)

    
    #####################
    ### Prepare Model ###
    #####################


    for f in os.listdir('/autofs/space/somes_002/users/pweiser/FittingChallenge2024/models/'+exp):
        if f.startswith("src"):
            sys.path.insert(0, '/autofs/space/somes_002/users/pweiser/FittingChallenge2024/models/'+exp+'/')
            sys.path.insert(0, '/autofs/space/somes_002/users/pweiser/FittingChallenge2024/models/'+exp+'/'+f)
            
            from config import params
            from src.initialize import initialize_model_folder, my_copy
            from src.training import training, validation
            from src.model import yModel, uModel
            from src.model2 import yyModel
            from src.resnet import resnet1d
            #from src.dataloader import SpecturmSampler, SpectrumDataset
            break

      
    model = yModel(nLayers=params["nLayers"], 
                    nFilters=params["nFilters"], 
                    dropout=0,
                    in_channels=params["in_channels"], 
                    out_channels=params["out_channels"]
                    )
    

    params["path_to_model"] = "/autofs/space/somes_002/users/pweiser/FittingChallenge2024/models/"+exp+"/"
    model.load_state_dict(torch.load(params["path_to_model"] + 'model_last.pt'))
    #model.load_state_dict(torch.load(params["path_to_model"] + 'model_best.pt'))
    model.to(device)


    #####################
    ### Remove Lipids ###
    #####################

    Data_LipidRemoved_rf, Data_Lipid_rf = runModelOnLipData(lip=lip, 
                                            lipProj=lipProj, 
                                            model=model,
                                            device=device)

    Data_LipidRemoved_rrrf = np.zeros(image_rrrf.shape, dtype=np.cfloat)
    Data_LipidRemoved_rrrf[headmask>0,:] = Data_LipidRemoved_rf.numpy()

    Data_Lipid_rrrf = np.zeros(image_rrrf.shape, dtype=np.cfloat)
    Data_Lipid_rrrf[headmask>0,:] = Data_Lipid_rf.numpy()

    return Data_LipidRemoved_rrrf, Data_Lipid_rrrf


def runModelOnLipData(lip, lipProj, model, device):

    spectra_energy = torch.sqrt(torch.sum(torch.abs(lip-lipProj)**2, dim=1))[:,None]
    lip /= spectra_energy
    lip = torch.stack((torch.real(lip), torch.imag(lip)), axis=1)
    lipProj /= spectra_energy
    lipProj = torch.stack((torch.real(lipProj), torch.imag(lipProj)), axis=1)

    prediction = torch.zeros(lip.shape, dtype=torch.cfloat)
    datasz = lipProj.shape[0]
    batchsz = 200
    
    loss_func = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for i in range(int(datasz/batchsz)+1):
            log = 'Percent: {:.2f}%'
            percent = (i+1)/int(datasz/batchsz)*100
            print(log.format(percent), end='\r')
            lip_batch = lip[i*batchsz:(i+1)*batchsz,:,:]
            lipProj_batch = lipProj[i*batchsz:(i+1)*batchsz,:,:]
            
            lip_batch, lipProj_batch = lip_batch.to(device), lipProj_batch.to(device)
            pred = model(lip_batch, lipProj_batch).cpu()
            prediction[i*batchsz:(i+1)*batchsz,:] = pred
    
    
    #loss = loss_func(prediction, true)
    #print('Pred loss: ', loss.item())
    #loss = loss_func(lip, true)
    #print('Inp loss: ', loss.item())
    prediction = prediction[:,0] + 1j*prediction[:,1]
    prediction = prediction*spectra_energy
    lip = lip[:,0] + 1j*lip[:,1]
    lip = lip * spectra_energy
    Data_LipidRemoved_rf = lip - prediction
    
    return Data_LipidRemoved_rf, prediction