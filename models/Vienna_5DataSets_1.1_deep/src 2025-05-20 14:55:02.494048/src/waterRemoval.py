import numpy as np
import matplotlib.pyplot as plt
import h5py
import nibabel as nib
import scipy.io
import time
import pandas as pd
import glob
import re
import os
import scipy.ndimage
import scipy.linalg
from pathlib import Path

os.chdir(Path(__file__).parent)

from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product

#Notes:
# v1 Test
# v2 Test
# v3 First Dataset

version='v_1.0'
path = '../data/'

#subjects = ['/3DMRSIMAP_Vol_04_A_1_2024-09-06_L2_0p0005', '3DMRSIMAP_Vol_08_A_1_2024-09-07_L2_0p0005'] #'3DMRSIMAP_Vol_10_A_1_2024-09-04_L2_0p0005','3DMRSIMAP_Vol_16_A_1_2024-09-05_L2_0p0005'
subjects = ['Vol5','Vol8','Vol9']

# Water Removal
b_RemWat = True
WatSuppComp = 32 # Number of component for the HSVD water removal (advised: 16 at 3T and 32 at 7T)
minFreq = -150 # -150Hz(7T) # +-0.5ppm
maxFreq = 150 # 150
parallel_jobs = 20

bandwidth = 2778 # Vienna: 2778 Paul: 3000
dwell_time = 1/bandwidth # Vienna 360000 nano seconds = 3.6*10⁻4 seconds

def HSVD(y, fs, k):
        N = len(y)
        L = int(np.floor(0.5*N))

        # Hankel Matric and SVD
        H = scipy.linalg.hankel(y[:L], y[L:N])
        u, s, vh = np.linalg.svd(H)

        # Compute Z prime
        Uk = u[:,:k]
        Ukt = Uk[1:,:]
        Ukb = Uk[:-1,:]
        Zp = np.matmul(np.linalg.pinv(Ukb),Ukt)

        # Compute poles
        w, v = np.linalg.eig(Zp)
        Z = np.matmul(np.matmul(np.linalg.inv(v), Zp), v)
        q = np.log(np.diag(Z))
        dt = 1/fs
        dampings = np.real(q)/dt
        dampings[dampings>10] = 10
        frequencies = np.imag(q)/(2*np.pi)/dt

        # Construct Basis
        t = np.arange(start=0,stop=len(y)*dt,step=dt)
        basis = np.exp(np.matmul(t[:,None], (dampings+2*np.pi*1j*frequencies)[None,:]))

        # Compute the amplitude estimates
        amps = np.matmul(np.linalg.pinv(basis),y)
        return frequencies, dampings, basis, amps

def WaterSuppressionWrapper(image_rrrt, mask, fs, k, minFreq, maxFreq):
        global WaterSuppression
        def WaterSuppression(tup):
            (x,y,z)=tup
            fid = image_rrrt[x,y,z,:] 
            if mask[x,y,z]:
                #sta_time = time.time()
                frequencies, dampings, basis, amps = HSVD(y = fid,
                                                        fs = fs,
                                                        k = k)
                indx = np.where(np.logical_and(frequencies >= minFreq, maxFreq >= frequencies))[0]
                filtFid = fid - np.sum(np.matmul(basis[:,indx], np.diag(amps[indx])), 1)

                return (filtFid, x,y,z, np.sum(np.matmul(basis[:,indx], np.diag(amps[indx])), 1))
        return WaterSuppression

for sub in subjects:
    p_mask= path + sub + '/masks/brain_mask.npy'
    p_cc = path + sub + '/OriginalData/data.npy' # coil combined reconstructed data
    p_scalp_mask = path + sub + '/masks/lipid_mask.npy'
    p_save = path + sub + '/OriginalData/'

    #####################
    ##### Load Data #####
    #####################

    brainmask = np.load(p_mask)  #Hauke: That should be the brain mask

    csi_rrrt = np.load(p_cc)

    skmask = np.load(p_scalp_mask)  # scalp mask / lipid mask

    headmask = brainmask + skmask

    #########################
    ##### Water Removal #####
    #########################


    if b_RemWat:
        print("####### Water Suppression #######")
        sta_time = time.time()

        image_grid = np.array(csi_rrrt)
        s = image_grid.shape
        image_rrrt = np.zeros(image_grid.shape, dtype=np.complex64)
        water_rrrt = np.zeros(image_grid.shape, dtype=np.complex64)


        WaterSuppression = WaterSuppressionWrapper(image_rrrt = image_grid, 
                                                    mask = headmask,
                                                    fs = 1/dwell_time, 
                                                    k = WatSuppComp, 
                                                    minFreq = minFreq, 
                                                    maxFreq = maxFreq)

        all_sl = list(range(s[2]))
        i=0
        slices = []
        cur_tup = []
        while i < s[2]:
            if len(cur_tup) < 3-1 and i<s[2]-1:
                cur_tup.append(all_sl[i])
            else:
                cur_tup.append(all_sl[i])
                slices.append(tuple(cur_tup))
                cur_tup = []
            i+=1

        print("all slices: ", slices)
        #slices=[(20,)]
        for sl in slices:
            print("Slice: ", sl)
            res = Parallel(n_jobs=parallel_jobs)(delayed(WaterSuppression)(tup=tup)
                                for tup in tqdm(product(range(s[0]), range(s[1]), sl), total=s[0]*s[1]*len(sl), position=0, leave=True))

            for tup in res:
                if tup is not None:
                    filtFid, x, y, z, waterFid= tup
                    image_rrrt[x,y,z] = filtFid
                    water_rrrt[x,y,z] = waterFid

        sto_time = time.time()
        print('Water Removal: ', sto_time-sta_time)

        np.save(p_save + 'IsolatedWater.npy', water_rrrt)
        np.save(p_save + 'SupressedWater.npy',  image_rrrt)


    
        


