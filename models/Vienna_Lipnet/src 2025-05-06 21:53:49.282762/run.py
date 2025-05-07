import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # or whichever GPU you want
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import random
import h5py
#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from torchsummary import summary
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from config import params
from src.initialize import initialize_model_folder, my_copy, read_args
from src.training import training, validation
from src.model import yModel
from src.dataloader import SpectrumDatasetLoad

##################
### Initialize ###
##################


params = read_args(params)
print(params["model_name"])

initialize_model_folder(params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using decive: ", device)




##########################
### Model & Dataloader ###
##########################

train_dataset = SpectrumDatasetLoad(params=params,
                                    files=params["train_subjects"],
                                    version=params["data_version"],
                                    aug=True)

val_dataset = SpectrumDatasetLoad(params=params,
                                    files=params["val_subjects"],
                                    version=params["data_version"],
                                    aug=False)

train_dataloader = DataLoader(train_dataset,
                            num_workers=params["num_worker"],
                            shuffle=True,
                            batch_size=params["batch_size"])

val_dataloader = DataLoader(val_dataset,
                            num_workers=params["num_worker"],
                            shuffle=True,
                            batch_size=params["batch_size"])


model = yModel(nLayers=params["nLayers"], 
                nFilters=params["nFilters"], 
                dropout=params["dropout"],
                in_channels=params["in_channels"], 
                out_channels=params["out_channels"]
                ).to(device)


if params['preload']:
    print('loading model ' + params['preload_model'])
    model.load_state_dict(torch.load('models/' + params['preload_model'] + '/model_last.pt', map_location=device))


params["loss_func"] = nn.MSELoss()



params["optimizer"] = Adam(model.parameters(), lr=params["lr"])
params["scheduler"] = MultiStepLR(params["optimizer"], milestones=params["milestones"], gamma=params["gamma"])

################
### Training ###
################


if params["train"]:
    best_loss = 0
    f = open(params["path_to_model"] + "loss.txt", "a")
    f.write('Epoch; Epoch Loss; Validation Loss; Learning Rate; ')
    f.write('\n')
    f.close()
    for epoch in range(params["epochs"]):
        model, train_loss = training(model=model, 
                                    params=params, 
                                    dataloader=train_dataloader, 
                                    device=device,
                                    epoch=epoch)
        
        val_loss = validation(model=model, 
                            params=params, 
                            dataloader=val_dataloader, 
                            device=device,
                            epoch=epoch)
        
        f = open(params["path_to_model"] + "loss.txt", "a")
        log = 'Epoch: {:03d}, Loss: {:.10f}, Val Loss: {:.10f}, LR: {:.10f}'
        f.write(log.format(epoch+1, train_loss, val_loss, params["scheduler"].get_last_lr()[0]))
        
        torch.save(model.state_dict(), params["path_to_model"] + "model_last.pt")
        if val_loss < best_loss or best_loss == 0:
            best_loss = val_loss
            torch.save(model.state_dict(), params["path_to_model"] + "model_best.pt")
            f.write(', best model')
        params["scheduler"].step()
        f.write('\n')
        f.close()



##################
### Prediction ###
##################

if params["predict"]:

    ### Load Model ###
    params["path_to_model"] = "/autofs/space/somes_002/users/pweiser/lipidSuppressionCNN/models/"+exp+"/"
    model.load_state_dict(torch.load(params["path_to_model"] + 'model_last.pt'))
    #model.load_state_dict(torch.load(params["path_to_model"] + 'model_best.pt'))

    prediction(params, model)
        
