import os
import sys
import numpy as np
import nibabel as nib
import h5py
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import re
import glob



def simulation(nSpectra=60, sub='Sub1'):
    p = '/autofs/space/somes_002/users/pweiser/FittingChallenge2024/data/'+sub+'/'+sub+'_all.mat'
    fh = h5py.File(p, 'r')

    metab_basis = np.array(fh['VtMeta'])
    metab_basis = metab_basis['real'] + 1j* metab_basis['imag']
    fh.close()

    metabolites = ['Asp', 'GABA', 'Gln', 'Glu', 'Lac', 'Lac41', 'NAA', 'NAAG', 'PE', 'PE40', 'Tau', 'mIns', 'sIno', 'tCho', 'tCr', 'tCr39']

    # FYI: I put the same values for 'Lac41', 'PE40', 'tCr39' as for the regular metabolite groups.
    # Should those metabolites be linked?
    mean_values = np.array([3.0, 2.0, 2.0, 8.0, 0.5, 0.5, 10.0, 1.0, 0.7, 0.7, 1.5, 6.5, 0.35, 1.5, 6.0, 6.0])
    std_values = np.array([2.0, 1.5, 1.5, 6.5, 5.0, 5.0, 7.0, 0.5, 0.5, 0.5, 1.0, 2.5, 0.15, 1.0, 5.5, 5.5])

    ### Setting parameters ###
    N=384
    NbM = len(metabolites)

    NMRFreq= 127.7 *10**6 #297189866.0 #

    dwell_time = 8.3*10**(-4)  # 0.83ms
    sampling_rate = 1/dwell_time  # Hz  sampling_rate = 1/dwell_time

    t = (np.arange(N+2)[2:] / sampling_rate)
    metab_basis = np.transpose(metab_basis[2:,:])

    ### Simulation parameters ###
    Amplitude = mean_values[None,:,None] * np.random.randn(nSpectra, NbM, 1) + std_values[None,:,None]
    Amplitude = Amplitude.clip(min=0)

    MaxAcquDelay=0.002
    AcquDelay = (np.random.rand(nSpectra, 1,1)-0.5)*2 * MaxAcquDelay

    PhShift=np.random.rand(nSpectra, NbM, 1) * 2 * np.pi
    TimeSerieClean = np.zeros(( N), dtype=np.complex64)

    MaxFreq_Shift = 40
    FreqShift = (np.random.rand(nSpectra, 1)*2 - 1) * MaxFreq_Shift

    MinPeak_Width=2.5
    MaxPeak_Width=7.5
    PeakWidth = MinPeak_Width + np.random.rand(nSpectra, 1) * (MaxPeak_Width - MinPeak_Width)
    PeakWidth_Gau = PeakWidth

    MinPeak_Width=22.5
    MaxPeak_Width=27.5
    PeakWidth = MinPeak_Width + np.random.rand(nSpectra, 1) * (MaxPeak_Width - MinPeak_Width)
    PeakWidth_Lor = PeakWidth

    ### Simulation of Spectra ###
    MetabData_smt = metab_basis[None,:] * np.exp(2 * np.pi * 1j * (t[None,None,:] + AcquDelay))
    MetabData_smt = Amplitude * MetabData_smt * np.exp(1j * PhShift)  
    MetabData_st = np.sum(MetabData_smt, axis=1)

    MetabData_st *= np.exp( (t[None,:]* 1j * 2 * np.pi * FreqShift ) + (- (np.square(t[None,:])) * (np.square(PeakWidth_Gau))) + ((np.absolute(t[None,:])) * (- PeakWidth_Lor) ) )
    MetabData_sf = np.fft.fftshift(np.fft.fft(np.conj(MetabData_st), axis=-1), axes=-1)

    spectra_energy = np.mean(np.sqrt(np.sum(np.abs(MetabData_sf)**2, axis=-1)))
    MetabData_sf/=spectra_energy

    return MetabData_sf


def simulation2(nSpectra=60):
    MaxSNR=8 # not used
    MinSNR=1 # not used

    N=384
    NMRFreq= 127.7 *10**6 #297189866.0 #

    dwell_time = 8.3*10**(-4)  # 0.83ms
    sampling_rate = 1/dwell_time  # Hz  sampling_rate = 1/dwell_time

    # Calculate the time vector
    t = np.arange(N) / sampling_rate

    # Simulation parameters
    MaxAcquDelay=0.002
    AcquDelay = (np.random.rand(nSpectra, 1)-0.5)*2 * MaxAcquDelay

    PhShift=np.random.rand(nSpectra, 1) * 2 * np.pi
    TimeSerieClean = np.zeros(( N), dtype=np.complex64)

    MaxFreq_Shift = 40
    FreqShift = (np.random.rand(nSpectra, 1)*2 - 1) * MaxFreq_Shift

    #MinPeak_Width=10#4
    #MaxPeak_Width=35#20
    #PeakWidth = MinPeak_Width + np.random.rand(nSpectra, 1) * (MaxPeak_Width - MinPeak_Width)
    #ponder_peaks = np.random.rand(nSpectra, 1)
    #PeakWidth_Gau = np.multiply(ponder_peaks, PeakWidth)
    #PeakWidth_Lor = np.multiply(1-ponder_peaks, PeakWidth)

    MinPeak_Width=4
    MaxPeak_Width=20
    PeakWidth = MinPeak_Width + np.random.rand(nSpectra, 1) * (MaxPeak_Width - MinPeak_Width)
    ponder_peaks = np.random.rand(nSpectra, 1)
    PeakWidth_Gau = np.multiply(ponder_peaks, PeakWidth)

    MinPeak_Width=15
    MaxPeak_Width=45
    PeakWidth = MinPeak_Width + np.random.rand(nSpectra, 1) * (MaxPeak_Width - MinPeak_Width)
    ponder_peaks = np.random.rand(nSpectra, 1)
    PeakWidth_Lor = np.multiply(1-ponder_peaks, PeakWidth)


    SNR = MinSNR + np.random.rand(nSpectra, 1) * (MinSNR - MinSNR)



    ########################################
    #### Perpare metabolite information ####
    ########################################
    def ReadModeFiles(index,list_file):
        NbM = len(list_file)
        temp_modes = [None] * NbM  # empty list of size NbM 
        for i, filename in enumerate(list_file):
            metabo_mode = pd.read_csv(filename, header=None, skiprows=[0]).values
            m = re.search("[0-9]T_.{1,6}_Exact", filename)
            name = bytes(filename[m.span()[0]+3:m.span()[1]-6].strip(), 'utf8')
            temp_modes[index[name]] = metabo_mode
        return temp_modes

    index = {}  # mapping of metabolite name to index
    mean_std_csv = pd.read_csv('/autofs/space/somes_002/users/pweiser/FittingChallenge2024/'+'/MetabModes/Metab_Mean_STD.txt', header=None).values

    for i, v in enumerate(mean_std_csv[:, 0].astype(str)):
        index[ bytes(v.strip(), 'utf8') ] = i

    mean_std = mean_std_csv[:, 1:].astype(np.float32)

    list_file = glob.glob('/autofs/space/somes_002/users/pweiser/FittingChallenge2024/'+'/MetabModes/3T_TE0/*Exact_Modes.txt')
    NbM = len(list_file)
    metabo_modes = [[[None] for j in range(NbM)] for i in range(6)]
    temp_modes = ReadModeFiles(index,list_file)
    metabo_modes[0]=temp_modes

    # Metabolic parameters
    TempMetabData = np.zeros( (len(metabo_modes[0]), N), dtype=np.complex64)
    BasisI = 0#np.floor(np.random.rand(nSpectra, 1) * 6)

    Amplitude = mean_std[:, 1]* np.random.randn(nSpectra, NbM) + mean_std[:, 0]
    Amplitude = Amplitude.clip(min=0)


    #############################
    #### Simulate metabolite ####
    #############################

            
    MetabSpectrum = np.zeros((nSpectra, N), dtype=np.complex128)
    for n in tqdm(range(nSpectra)):

        TempMetabData =0*TempMetabData
        for f, mode in enumerate(metabo_modes[0]):   # metabo_modes[int(BasisI[n])]
                Freq = ((4.7-mode[:, 0]) * 1e-6 * NMRFreq)[...,None]

                for Nuc in range(len(Freq)):
                    if (mode[Nuc, 0] > 0.0) & (mode[Nuc, 0] < 4.5)  : # only for the window of interest 
                        TempMetabData[f, :] += mode[Nuc, 1][...,None] * np.exp(1j * mode[Nuc, 2][...,None]) * np.exp(2 * np.pi * 1j * (t + AcquDelay[n])  * (Freq[Nuc]))

        TimeSerieClean=0*TimeSerieClean
        for f, _ in enumerate(metabo_modes[0]):  # metabo_modes[int(BasisI[ex])]
            TimeSerieClean[:] += Amplitude[n, f] * TempMetabData[f, :]* np.exp(1j * PhShift[n])  

        TimeSerieClean[:] *= np.exp( (t* 1j * 2 * np.pi * FreqShift[n] ) + (- (np.square(t)) * (np.square(PeakWidth_Gau[n]))) + ((np.absolute(t)) * (- PeakWidth_Lor[n]) ) )
        SpectrumTemp = np.fft.fftshift(np.fft.fft(TimeSerieClean[:],axis=0))

        # SNR
        #NCRand=(np.random.randn(N) + 1j * np.random.randn(N))
        #TimeSerie = TimeSerieClean #+ np.fft.ifft(SpectrumTemp.std()/0.65 / SNR[n] * NCRand,axis=0)
        
        MetabSpectrum[n] = np.fft.fftshift(np.fft.fft(TimeSerieClean))
    spectra_energy = np.sqrt(np.sum(np.abs(MetabSpectrum)**2, axis=-1))
    MetabSpectrum/=np.mean(spectra_energy)

    return MetabSpectrum
        