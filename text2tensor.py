#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys 
import torch

def tensor_from_txt(f):
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   obs = list()
   for line in f:
      row = [float(x) for x in line.split()]
      obs.append(torch.tensor(row, device=device))
   return torch.cat(obs).view(len(obs),-1)

if __name__ == '__main__':
   if len(sys.argv) > 1:
      with open(sys.argv[1]) as f:
         x = tensor_from_txt(f)
   else:
      x = tensor_from_txt(sys.stdin)
   torch.save(x, sys.stdout)
