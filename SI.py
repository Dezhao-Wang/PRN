import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable
import PRNet as prn

class SI(nn.Module):
	def __init__(self, nFeats = 64,nDenselayer = 6, growthRate = 32,dsFlag = False):
		super(SI,self).__init__()
		self.dsFlag = dsFlag
		self.preConv = nn.Conv2d(1,nFeats,kernel_size = 3, padding = 1, bias = False)
		
		self.PRB0 = prn.PRB(nFeats, nDenselayer,growthRate,0)
		self.PRB1 = prn.PRB(nFeats,nDenselayer,growthRate,self.PRB0)
	
		self.Conv = nn.Conv2d(nFeats,nFeats,kernel_size = 1, padding = 0, bias = True)

	def forward(self, x):

		if self.dsFlag is False:
			res = self.preConv(x)
			x = self.PRB0(res)
			x = self.PRB1(x)
			x = self.Conv(x)
			x = res + x
			return x