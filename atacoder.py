#=/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.distributions.constraints import positive

from atachic_data import ATACHiCData

ISZ = 1275
OSZ = 50
NN = 100

def InverseLinear(x):
   # Inverse-Linear activation function
   return 1.0 / (1.0-x+torch.abs(x)) + x+torch.abs(x)

class Encoder(nn.Module):

   def __init__(self):
      super(Encoder, self).__init__()
      self.lyrE1 = nn.Sequential(
         nn.Linear(ISZ, NN),
         #nn.Dropout(),
         nn.BatchNorm1d(NN),
         nn.ReLU()
      )
      self.lyrE2 = nn.Sequential(
         nn.Linear(NN, NN),
         #nn.Dropout(.2),
         nn.BatchNorm1d(NN),
         nn.ReLU(),
         nn.Linear(NN, NN),
         #nn.Dropout(),
         nn.BatchNorm1d(NN),
      )
      self.a = nn.Linear(NN,OSZ)
      self.b = nn.Linear(NN,OSZ)

   def forward(self, x): 
      x = self.lyrE1(x)
      x = torch.relu(x + self.lyrE2(x))
      a = torch.clamp(InverseLinear(self.a(x)/2), min=.01, max=100)
      b = torch.clamp(InverseLinear(self.b(x)/2), min=.01, max=100)
      return (a,b)

   def optimize(self, train_data, test_data, epochs=30, bsz=256):
      # The initial learning rates are set to avoid the parameters
      # to blow up. If they are higher no learning takes place.
      optimizer = \
            torch.optim.Adam(self.parameters(), lr=0.001)
      sched = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [29])

      batches = DataLoader(dataset=train_data,
            batch_size=bsz, shuffle=True)
      test_set = DataLoader(dataset=test_data,
            batch_size=bsz, shuffle=True)

      best = float('inf')

      for ep in range(epochs):
         batch_loss = 0.0
         self.train()
         for bno,data in enumerate(batches):
            atac = torch.clamp(data[:,:50], min=.001, max=.9999)
            hic = data[:,50:]
            # Shrink 5% of the entries.
            shrink = torch.ones_like(hic, device=data.device)
            idx = torch.rand(shrink.shape, device=data.device) < .05
            shrink[idx] = torch.rand(shrink.shape, device=data.device)[idx]
            # Random factor.
            #rfact = .8 + .4 * torch.rand(1, device=data.device)
            (a,b) = self(hic * shrink)
            #(a,b) = self(hic)
            loss = -torch.mean(Beta(a,b).log_prob(atac))
            batch_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

         sched.step()

         # Test data.
         self.eval()
         with torch.no_grad():
            test_rcst = 0.0
            for sno,data in enumerate(test_set):
               hic = data[:,50:]
               atac = torch.clamp(data[:,:50], min=.001, max=.9999)
               (a,b) = self(hic)
               test_rcst -= float(torch.mean(Beta(a,b).log_prob(atac)))

         # Print logs on stderr.
         if test_rcst / sno < best: best = test_rcst / sno
         sys.stderr.write('%d\t%f\t%f\t%f\n' % \
               (ep, batch_loss / bno, test_rcst / sno, best))


# MD5 sums
# e720e20f75fc2d80142d8dee668bdd13  train_dual_13.tch
# 9fe500b6559c218cc1f163cb7a517b2c  test_dual_Xcas.tch
# 1493f06a1e8a77714c1451ff6ec5dd47  test_dual_Xmus.tch
train_data = torch.load('train_dual_13.tch')
test_data = torch.load('test_dual_Xcas.tch')
torch.manual_seed(123)
E = Encoder().cuda()
E.optimize(train_data, test_data)
torch.save(E, 'model_atacoder.tch')
test_data = torch.load('test_dual_Xmus.tch')
#E = torch.load('model_atacoder.tch')
atac = test_data[:,:50]
hic = test_data[:,50:]
E.eval()
a,b = E(hic)
ab = a / (a+b)
for i in range(len(test_data)):
   print '%f\t%f' % (atac[i,24], ab[i,24])
