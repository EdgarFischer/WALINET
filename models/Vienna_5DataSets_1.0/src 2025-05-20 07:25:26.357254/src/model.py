import numpy as np
import h5py
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.parameter import Parameter



class uModel(nn.Module):
    def __init__(self, nLayers, nFilters, dropout, in_channels, out_channels):
        super().__init__()
        self.nLayers = nLayers
        self.nFilters = nFilters
        self.kernel_size = 3
        self.out_channels = out_channels
        self.in_channels = 2*in_channels
        self.dropout = dropout
        self.pool_size = 2
        self.multFilter = [2**i for i in range(self.nLayers+1)]

        
        self.encoder = nn.ModuleList([DownBlock(in_channels = int(self.in_channels*self.nFilters*self.multFilter[i]), 
                                            out_channels = self.in_channels*self.nFilters*self.multFilter[i+1], 
                                            kernel_size = self.kernel_size,
                                            pool_size=self.pool_size,
                                            dropout=self.dropout
                                           ) for i in range(self.nLayers)])
        
        self.decoder = nn.ModuleList([UpBlock(in_channels = out_channels*self.nFilters*self.multFilter[i+1] + int(self.in_channels*self.nFilters*self.multFilter[i]), 
                                            out_channels = int(out_channels*self.nFilters*self.multFilter[i]), 
                                            kernel_size = self.kernel_size,
                                            dropout=self.dropout
                                           ) for i in range(self.nLayers)])
        
        self.bottleconv = ConvBlock(in_channels=self.in_channels*self.nFilters*self.multFilter[-1], 
                                    out_channels=out_channels*self.nFilters*self.multFilter[-1], 
                                    kernel_size=self.kernel_size,
                                    dropout=self.dropout)
        
        self.initconv=ConvBlock(in_channels=self.in_channels, 
                                out_channels=self.in_channels*self.nFilters*self.multFilter[0], 
                                kernel_size=self.kernel_size,
                                dropout=0,
                                batchnorm=False)
        
                     
        self.finconv1=ConvBlock(in_channels=int(out_channels*self.nFilters*self.multFilter[0]) + self.in_channels + self.in_channels*self.nFilters*self.multFilter[0], 
                                out_channels=int(out_channels*self.nFilters*self.multFilter[0]), 
                                kernel_size=self.kernel_size,
                                dropout=self.dropout,
                                batchnorm=False,
                                mid_channels='in')
        
        self.finconv2=ConvBlock(in_channels=int(out_channels*self.nFilters*self.multFilter[0]) + self.in_channels + self.in_channels*self.nFilters*self.multFilter[0], 
                                out_channels=out_channels, 
                                kernel_size=1,
                                dropout=0,
                                batchnorm=False,
                                mid_channels='in',
                                activation=False)
        
        
    def forward(self,x1,y1):

        #Initial Block
        x1 = F.pad(x1, (8,8), 'replicate') 
        y1 = F.pad(y1, (8,8), 'replicate') 
        x=torch.cat((x1,y1), dim=1)
        x = self.initconv(x)

        # Encoder
        allx = [x]
        for i in range(self.nLayers):
            allx.append(self.encoder[i](allx[-1]))
        
        # Bottleneck
        out=allx[-1]
        out=self.bottleconv(out)

        # Decoder
        for i in range(self.nLayers):
            out = self.decoder[-i-1]((out, allx[-i-2]))
        
        # Final Block
        out = torch.cat((out,x1,y1,x),dim=1)
        out = self.finconv1(out)
        out = torch.cat((out,x1,y1,x),dim=1)
        out = self.finconv2(out)
        return out[:,:,8:-8]



class yModel(nn.Module):
    def __init__(self, nLayers, nFilters, dropout, in_channels, out_channels):
        super().__init__()
        self.nLayers = nLayers
        self.nFilters = nFilters
        self.kernel_size = 3
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dropout = dropout
        self.pool_size = 2
        self.multFilter = [2**i for i in range(self.nLayers+1)]
        self.b = 16
        #self.multFilter = [1/self.nFilters]+[2**(i+1) for i in range(self.nLayers)]
        #self.multFilter = [int(np.ceil(i/2+2)) for i in range(self.nLayers+1)]
        #self.multFilter = [2, 4, 6, 8, 10]
        #self.multFilter = [1/self.nFilters, 4, 5, 6, 7] # this is acutally working!
        #self.multFilter = [4, 5, 5, 6, 6]
        #print('Multiplicator for filters: ', self.multFilter)

        
        self.encoder1 = nn.ModuleList([DownBlock(in_channels = int(in_channels*self.nFilters*self.multFilter[i]), 
                                            out_channels = in_channels*self.nFilters*self.multFilter[i+1], 
                                            kernel_size = self.kernel_size,
                                            pool_size=self.pool_size,
                                            dropout=self.dropout
                                           ) for i in range(self.nLayers)])
        self.encoder2 = nn.ModuleList([DownBlock(in_channels = int(in_channels*self.nFilters*self.multFilter[i]), 
                                            out_channels = in_channels*self.nFilters*self.multFilter[i+1], 
                                            kernel_size = self.kernel_size,
                                            pool_size=self.pool_size,
                                            dropout=self.dropout
                                           ) for i in range(self.nLayers)])
        self.decoder = nn.ModuleList([UpBlock(in_channels = out_channels*self.nFilters*self.multFilter[i+1] + int(2*out_channels*self.nFilters*self.multFilter[i]), 
                                            out_channels = int(out_channels*self.nFilters*self.multFilter[i]), 
                                            kernel_size = self.kernel_size,
                                            dropout=self.dropout
                                           ) for i in range(self.nLayers)])
        
        self.bottleconv = ConvBlock(in_channels=2*in_channels*self.nFilters*self.multFilter[-1], 
                                    out_channels=out_channels*self.nFilters*self.multFilter[-1], 
                                    kernel_size=self.kernel_size,
                                    dropout=self.dropout)
        
        self.initconv1=ConvBlock(in_channels=in_channels, 
                                out_channels=in_channels*self.nFilters*self.multFilter[0], 
                                kernel_size=self.kernel_size,
                                dropout=0,
                                batchnorm=False)
        self.initconv2=ConvBlock(in_channels=in_channels, 
                                out_channels=in_channels*self.nFilters*self.multFilter[0], 
                                kernel_size=self.kernel_size,
                                dropout=0,
                                batchnorm=False)
                     
        self.finconv1=ConvBlock(in_channels=int(out_channels*self.nFilters*self.multFilter[0]) + 2*in_channels + 2*in_channels*self.nFilters*self.multFilter[0], 
                                out_channels=int(out_channels*self.nFilters*self.multFilter[0]), 
                                kernel_size=self.kernel_size,
                                dropout=self.dropout,
                                batchnorm=False,
                                mid_channels='in')
        
        self.finconv2=ConvBlock(in_channels=int(out_channels*self.nFilters*self.multFilter[0]) + 2*in_channels + 2*in_channels*self.nFilters*self.multFilter[0], 
                                out_channels=out_channels, 
                                kernel_size=1,
                                dropout=0,
                                batchnorm=False,
                                mid_channels='in',
                                activation=False)
        
        
    def forward(self,x1,y1):
        #Initial Block
        x1 = F.pad(x1, (self.b,self.b), 'replicate') 
        y1 = F.pad(y1, (self.b,self.b), 'replicate') 
        x2 = self.initconv1(x1)
        y2 = self.initconv2(y1)

        # Encoder
        allx = [x2]
        ally = [y2]
        for i in range(self.nLayers):
            allx.append(self.encoder1[i](allx[-1]))
            ally.append(self.encoder2[i](ally[-1]))
        
        # Bottleneck
        out=torch.cat((allx[-1], ally[-1]), dim=1)
        out=self.bottleconv(out)
        # Decoder
        for i in range(self.nLayers):
            out = self.decoder[-i-1]((out, allx[-i-2], ally[-i-2]))
        
        # Final Block
        out = torch.cat((out,x1,y1,x2,y2),dim=1)
        out = self.finconv1(out)
        out = torch.cat((out,x1,y1,x2,y2),dim=1)
        out = self.finconv2(out)
        return out[:,:,self.b:-self.b]



        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, batchnorm=False, mid_channels='out', activation=True):
        super().__init__()
        self.bn = batchnorm
        #self.drop1=nn.Dropout(p=dropout)
        self.drop2=nn.Dropout(p=dropout)
        if mid_channels=='in':
            mchannels=in_channels
        elif mid_channels=='out':
            mchannels=out_channels
        self.activation=activation
        

        self.conv1=nn.Conv1d(in_channels=in_channels, 
                       out_channels=mchannels, 
                       kernel_size=kernel_size, 
                       stride=1, 
                       padding='same')
        self.conv1.bias.data.fill_(0.0)
        #if self.bn:
        #    self.batchnorm1=nn.BatchNorm1d(out_channels, momentum=0.03)
        self.act1=nn.PReLU(mchannels)

        self.conv2=nn.Conv1d(in_channels=mchannels, 
                       out_channels=out_channels, 
                       kernel_size=kernel_size, 
                       stride=1, 
                       padding='same')
        self.conv2.bias.data.fill_(0.0)
        #if self.bn:
        #    self.batchnorm2=nn.BatchNorm1d(out_channels, momentum=0.03)
        self.act2=nn.PReLU(out_channels)
        
    def forward(self, x):
        
        
        x=self.conv1(x)
        x=self.act1(x)
        #x=self.drop1(x)
        #if self.bn:
        #    x=self.batchnorm1(x)
        x=self.conv2(x)
        if self.activation:
            x=self.act2(x)
            x=self.drop2(x)
        #if self.bn:
        #    x=self.batchnorm2(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout):
        super().__init__()
        
        self.conv = ConvBlock(in_channels=in_channels, 
                         out_channels=out_channels, 
                         kernel_size=kernel_size,
                         dropout=dropout)
        self.maxPool=nn.MaxPool1d(kernel_size=pool_size,#kernel_size, 
                             stride=None, 
                             padding=0)
    
    def forward(self,x):
        #print("down1: ", x.shape)
        x=self.maxPool(x)
        #print("down2: ", x.shape)
        x=self.conv(x)
        #print("down3: ", x.shape)
        return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        
        self.conv = ConvBlock(in_channels=in_channels, 
                         out_channels=out_channels, 
                         kernel_size=kernel_size,
                         dropout=dropout)
        self.upsample = nn.Upsample(scale_factor=2)
    
    def forward(self,tup):
        #x,y,z=tup
        #print('upx1: ', x.shape)
        #print('upy: ', y.shape)
        #print('upz: ', z.shape)
        x=self.upsample(tup[0])
        #print('upx2: ', ((x,)+tup[1:]).shape)
        xyz=torch.cat((x,)+tup[1:],dim=1)
        #print('upxyz: ', xyz.shape)
        xyz=self.conv(xyz)
        return xyz