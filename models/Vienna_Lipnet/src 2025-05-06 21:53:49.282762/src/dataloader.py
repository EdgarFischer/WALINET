import numpy as np
import h5py
import random
from random import sample 
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SpectrumDatasetLoad(Dataset):
    def __init__(self, params, files, version, aug=True):
        self.params = params
        self.files = files
        self.version=version
        self.aug=aug
        
            

        print("Loading Data:")
        
        for i, sub in enumerate(self.files):
            print("++ " + sub + ' ++')

            s=10000
            fh = h5py.File(params["path_to_data"] + sub + '/TrainData/'+'TrainData_'+self.version+'.h5', 'r')
            spectra = torch.tensor(np.array(fh['spectra'][:]), dtype=torch.cfloat)
            lipid = torch.tensor(np.array(fh['lipid'][:]), dtype=torch.cfloat)
            if 'water' in fh.keys():
                water = torch.tensor(np.array(fh['water'][:]), dtype=torch.cfloat)
                nuisance = lipid #+ water ACHTUNG HABE ICH HART GEÄNDERT HAUKE
            else:
                nuisance = lipid
            nuisance_proj = torch.tensor(np.array(fh['lipid_proj'][:]), dtype=torch.cfloat)
            
            idx = np.any(np.isnan(np.array(spectra)), axis=1)
            spectra = spectra[~idx,:]
            nuisance = nuisance[~idx,:]
            nuisance_proj = nuisance_proj[~idx,:]
            
            if i==0:
                self.spectra = spectra
                self.nuisance = nuisance
                self.nuisance_proj = nuisance_proj
            else:
                self.spectra = torch.cat((self.spectra, spectra), dim=0)
                self.nuisance = torch.cat((self.nuisance, nuisance), dim=0)
                self.nuisance_proj = torch.cat((self.nuisance_proj, nuisance_proj), dim=0)
            

        s1=0 #200
        s2=-1 #320
        self.spectra_energy = torch.sqrt(torch.sum(np.abs(self.spectra[:,s1:s2]-self.nuisance_proj[:,s1:s2])**2, dim=1))
        self.metab_energy = torch.sqrt(torch.sum(np.abs(self.spectra[:,s1:s2]-self.nuisance[:,s1:s2])**2, dim=1))
        self.metab_energy[self.metab_energy<0.25] = 0.25
        self.lip_max = torch.amax(np.abs(self.spectra), dim=-1)
        
        
        norm = self.spectra_energy #self.lip_max #self.metab_energy
        self.spectra /= norm[:,None]
        self.nuisance_proj /= norm[:,None]
        self.nuisance /= norm[:,None]
        self.metab_max = torch.amax(np.abs(self.spectra[:,s1:s2]-self.nuisance[:,s1:s2]), dim=-1)

        self.s = self.spectra.shape


    def __len__(self):
        return self.s[0]
    
    def __getitem__(self, index):

        in1 = self.spectra[index]
        in2 = self.nuisance_proj[index]
        out = self.nuisance[index] #self.metab_max[index]#
        spec_energy = self.spectra_energy[index]
        metab_energy = self.metab_energy[index]

        if self.aug:
            phase = torch.exp(2*np.pi*1j*torch.rand(1))
            scale = .5*torch.rand(1)+.75
            in1, in2, out = in1*phase*scale, in2*phase*scale, out*scale*phase
        

        in1 = torch.stack((torch.real(in1), torch.imag(in1)), dim=0)
        in2 = torch.stack((torch.real(in2), torch.imag(in2)), dim=0)
        out = torch.stack((torch.real(out), torch.imag(out)), dim=0)

        #in1 = torch.stack((torch.abs(in1), torch.angle(in1)), dim=0)
        #in2 = torch.stack((torch.abs(in2), torch.angle(in2)), dim=0)
        #out = torch.stack((torch.abs(out), torch.angle(out)), dim=0)

        return in1, in2, out, metab_energy, spec_energy
        


