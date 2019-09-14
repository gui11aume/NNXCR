#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

ISZ = 1275
DSZ = 50
NN = 120

class Encoder(nn.Module):

   def __init__(self):
      super(Encoder, self).__init__()
      self.lyrE1 = nn.Sequential(
         nn.Linear(ISZ, NN),
         nn.Dropout(.1),
         nn.BatchNorm1d(NN),
         nn.ReLU()
      )
      self.lyrE2 = nn.Sequential(
         nn.Linear(NN, NN),
         nn.Dropout(.1),
         nn.BatchNorm1d(NN),
         nn.ReLU(),
         nn.Linear(NN, NN),
         nn.Dropout(.1),
         nn.BatchNorm1d(NN),
      )   
      self.lyrE3 = nn.Linear(NN, DSZ)

   def forward(self, x): 
      x = self.lyrE1(x)
      x = torch.relu(x + self.lyrE2(x))
      return self.lyrE3(x)


class Decoder(nn.Module):

   def __init__(self):
      super(Decoder, self).__init__()
      self.lyrD1 = nn.Sequential(
         nn.Linear(DSZ, NN),
         nn.Dropout(.1),
         nn.BatchNorm1d(NN),
         nn.ReLU()
      )
      self.lyrD2 = nn.Sequential(
         nn.Linear(NN, NN),
         nn.Dropout(.1),
         nn.BatchNorm1d(NN),
         nn.ReLU(),
         nn.Linear(NN, NN),
         #nn.Dropout(.1),
         nn.BatchNorm1d(NN),
      )   
      self.lyrD3 = nn.Sequential(
         nn.Linear(NN, ISZ),
         nn.Softplus()
      )

   def forward(self, x): 
      x = self.lyrD1(x)
      x = torch.relu(x + self.lyrD2(x))
      return self.lyrD3(x)


class Checker(nn.Module):

   def __init__(self):
      super(Checker, self).__init__()
      self.lyrC1 = nn.Sequential(
         nn.Linear(DSZ, 128),
         nn.Dropout(.1),
         #nn.BatchNorm1d(128),
         nn.ReLU()
      )
      self.lyrC2 = nn.Sequential(
         nn.Linear(128, 128),
         nn.Dropout(.1),
         #nn.BatchNorm1d(128),
         nn.ReLU()
      )
      self.lyrC3 = nn.Linear(128, 1)

   def forward(self, x):
      x = self.lyrC1(x)
      x = self.lyrC2(x)
      return torch.sigmoid(self.lyrC3(x))


class AdversarialAutoEncoder(nn.Module):

   def __init__(self):
      super(AdversarialAutoEncoder, self).__init__()
      self.checker = Checker()
      self.encoder = Encoder()
      self.decoder = Decoder()

   def forward(self, x):
      z = self.encoder(x)
      x = self.decoder(z)
      return x, z

   def optimize(self, train_data, test_data, epochs=400, bsz=256):

      lossC = F.binary_cross_entropy
      lossR = nn.MSELoss()

      encoder_optimizer = \
            torch.optim.Adam(self.encoder.parameters(), lr=.01)
      decoder_optimizer = \
            torch.optim.Adam(self.decoder.parameters(), lr=.01)
      checker_optimizer = \
            torch.optim.Adam(self.checker.parameters(), lr=.01)

      encoder_sched = torch.optim.lr_scheduler.MultiStepLR(
            encoder_optimizer, [90, 330])
      decoder_sched = torch.optim.lr_scheduler.MultiStepLR(
            decoder_optimizer, [90, 330])
      checker_sched = torch.optim.lr_scheduler.MultiStepLR(
            checker_optimizer, [90, 330])

      batches = DataLoader(dataset=train_data,
            batch_size=bsz, shuffle=True)
      testset = DataLoader(dataset=test_data, batch_size=1024)

      for ep in range(epochs):
         b_rcst = 0.0
         b_fool = 0.0
         b_disc = 0.0
         self.train()
         for bno,data in enumerate(batches):
            data = data[:,50:]
            target_real = data.new_ones(len(data), 1)
            target_fake = data.new_zeros(len(data) ,1)
            target_disc = torch.cat((target_fake, target_real), 0)
            # Phase I: REGULARIZATION.
            # An important point is to tune the cost of the
            # regularization versus the reconstruction. This is
            # done by diluting the loss of the checker and the
            # fooler.
            with torch.no_grad():
               z_fake = self.encoder(data)
               z_real = torch.randn(len(data), DSZ, device=data.device)
            disc = self.checker(torch.cat((z_fake, z_real), 0))
            disc_loss = lossC(disc, target_disc) / 1. # <---
            b_disc += float(disc_loss)
            # Reset gradients.
            checker_optimizer.zero_grad()
            disc_loss.backward()
            # Update discriminator parameters
            checker_optimizer.step()
            # Phase II. RECONSTRUCTION.
            # Random factor.
            #rfact = .9 + .2 * torch.rand(data.shape, device=data.device)
            x, z = self(data)
            fool = self.checker(z)
            fool_loss = lossC(fool, target_real) / 1. # <---
            rcst_loss = lossR(x, data) / 100.
            loss = rcst_loss + fool_loss
            b_rcst += float(rcst_loss)
            b_fool += float(fool_loss)
            # Reset gradients.
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            checker_optimizer.zero_grad()
            loss.backward()
            # Update discriminator parameters
            encoder_optimizer.step()
            decoder_optimizer.step()

         encoder_sched.step()
         decoder_sched.step()
         checker_sched.step()

         s_fool = 0.
         s_rcst = 0.
         self.eval()
         for sno,data in enumerate(testset):
            data = data[:,50:]
            target_real = data.new_ones(len(data), 1)
            with torch.no_grad():
               x, z = self(data)
               rcst_loss = lossR(x, data) / 100.
               fool = self.checker(z)
               fool_loss = lossC(fool, target_real) / 1. # <---
            s_fool += float(fool_loss)
            s_rcst += float(rcst_loss)

         # Print logs on stderr.
         sys.stderr.write('%d\t%f\t%f\t%f\t%f\t%f\n' % \
               (ep, b_rcst/bno, s_rcst/sno, b_fool/bno,
                     s_fool/sno, b_disc/bno))


if __name__ == '__main__':
   train_data = torch.load('train_dual_13.tch')
   test_data = torch.load('test_dual_Xcas.tch')
   torch.manual_seed(123)
   AAE = AdversarialAutoEncoder().cuda()
   #AAE = torch.load('model_aae_hic_1.tch')
   AAE.optimize(train_data, test_data)
   torch.save(AAE, 'model_aae_hic.tch')
