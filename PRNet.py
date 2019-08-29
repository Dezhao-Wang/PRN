import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable
import SI

        
class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Progressive Rethinking Block
class PRB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate,PredPRB):
    super(PRB, self).__init__()
    nChannels_ = nChannels

    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(2*nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
    self.conv_feat_1x1 = nn.Conv2d(2*nChannels_, nChannels_, kernel_size=1, padding=0, bias=False)

    self.feature = 0
    self.PredPRB = PredPRB
    
  def forward(self, x):
    out = self.dense_layers(x)
    if self.PredPRB != 0:
        out = torch.cat((self.PredPRB.feature,out),1)
    else:
        out = torch.cat((out,out),1)
    self.feature = self.conv_feat_1x1(out) 
    out = self.conv_1x1(out)

    out = out + x
    return out


# Progressive Rethinking Network
class PRN(nn.Module):
    def __init__(self):
        super(PRN, self).__init__()
        nChannel = 1
        nDenselayer = 6
        nFeat = 64
        growthRate = 32
        self.nPRB = 10
        nPRB = self.nPRB

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # PRBs
        
        PRBs = []
        for i in range(0,nPRB):
            if i != 0:
                PRBs.append(PRB(nFeat,nDenselayer,growthRate,PRBs[i-1]))
            else:
                PRBs.append(PRB(nFeat,nDenselayer,growthRate,0))
        self.PRBs = nn.Sequential(*PRBs)

        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*nPRB, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
    
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv 
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)

        self.SI0 = SI.SI()
        self.SI1 = SI.SI()
        self.SI2 = SI.SI()
        self.SI3 = SI.SI()
    def forward(self, x,cuave0,cuave1,cuave2,cuave3):
        nPRB = self.nPRB
        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        cu0 = self.SI0(cuave0)
        cu1 = self.SI1(cuave1)
        cu2 = self.SI2(cuave2)
        cu3 = self.SI3(cuave3)
        F = [F_0]
        for i in range(0,nPRB):
            tmp = self.PRBs[i](F[i])
            if i == 1:
                tmp = tmp + cu0
            elif i == 3:
                tmp = tmp + cu1
            elif i == 5:
                tmp = tmp + cu2
            elif i == 7:
                tmp = tmp + cu3
            F.append(tmp)

        FF = F[1]
        for i in range(2,nPRB+1):
            FF = torch.cat((FF,F[i]),1)
        
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        output = self.conv_up(FDF)

        output = self.conv3(output)


        return output