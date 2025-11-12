import numpy as np
import h5py
import random
import time
import matplotlib.pyplot as plt 
import os
import sys

import torch
import torch.nn as nn

from config import params
from src.initialize import initialize_model_folder, my_copy
from src.training import training, validation
from src.model import yModel
from src.dataloader import SpecturmSampler, SpectrumDataset

device = torch.device(params["gpu"] if torch.cuda.is_available() else 'cpu')
#print("Using decive: ", device)




def prediction(params, model):
    model.to(device)
    model.eval()

    ### Load Data ###
    p_lip = params["path_to_data"] + params["val_subjects"][0] + '/' + params["val_subjects"][0] + '_LipStackEval_acc2.h5'
    p_lipProj = params["path_to_data"] + params["val_subjects"][0] + '/' + params["val_subjects"][0] + '_LipProjStackEval_acc2.h5'
    fh_lip = h5py.File(p_lip,'r')
    fh_lipProj = h5py.File(p_lipProj,'r')

    lip = torch.tensor(np.array(fh_lip['Lipid_Stack_File']), dtype=torch.cfloat)
    lipProj = torch.tensor(np.array(fh_lipProj['Lipid_Stack_File']), dtype=torch.cfloat)

    spectra_energy = torch.sqrt(torch.sum(torch.abs(lip-lipProj)**2, dim=1))[:,None]
    lip /= spectra_energy
    lip = torch.stack((torch.real(lip), torch.imag(lip)), axis=1)
    lipProj /= spectra_energy
    lipProj = torch.stack((torch.real(lipProj), torch.imag(lipProj)), axis=1)

    prediction = lipidRemoval(lip, lipProj, model)

    hf = h5py.File(params["path_to_model"] + 'predictions/' + params["val_subjects"][0] + '.h5', 'w')
    hf.create_dataset('pred', data=prediction)
    hf.close()

def lipidRemoval(lip, lipProj, model):

    ### Lipid Removal ###
    datasz = lipProj.shape[0]
    batchsz = 200
    sta_epoch = time.time()
    pred=None
    model.eval()
    with torch.no_grad():
        for i in range(int(datasz/batchsz)):
            log = 'Percent: {:.2f}%'
            percent = (i+1)/int(datasz/batchsz)*100
            print(log.format(percent), end='\r')
            lip_batch = lip[i*batchsz:(i+1)*batchsz,:,:]
            lipProj_batch = lipProj[i*batchsz:(i+1)*batchsz,:,:]
            
            lip_batch, lipProj_batch = lip_batch.to(device), lipProj_batch.to(device)
            pred = model(lip_batch, lipProj_batch).cpu()
            prediction[i*batchsz:(i+1)*batchsz,:] = pred[:,0] + 1j*pred[:,1]

    prediction = prediction*spectra_energy
    #lip = lip[:,0] + 1j*lip[:,1]
    #lip = lip * spectra_energy
    #lipProj = lipProj[:,0] + 1j*lipProj[:,1]
    #lipProj = lipProj * spectra_energy

    sto_epoch = time.time() - sta_epoch
    log_epoch = 'Lipid Removal: Time: {:.4f}'
    print(log_epoch.format(sto_epoch))

    return prediction


