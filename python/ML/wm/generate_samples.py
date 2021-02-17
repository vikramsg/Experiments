# first neural network with keras tutorial
import numpy as np

import pandas as pd

from skopt.space import Space
from skopt.sampler import Lhs

from tauGrad import *

"""
We sample u_LES and h from a range
and then use these to evaluate tau_w
which we use to train
"""

n_samples = 400  
space = Space([(2., 4.), (10., 100.)])

lhs = Lhs(lhs_type="classic", criterion=None)
x   = lhs.generate(space.dimensions, n_samples)

x_smpls = pd.DataFrame(data=x, columns=["h", "u"]) 
n_smpls = x_smpls.shape[0]

tau     = np.zeros(n_smpls)

nu  = 0.001
for it, i in enumerate(x):
  u   = i[0]
  h   = i[1]

  tau[it] = 10.*getTau(nu, h, u) 

  print(i, tau[it])

y  = pd.DataFrame(data=tau, columns=["tau"]) 

wm_data = pd.concat([x_smpls, y], axis=1)
wm_data.to_csv("wm_data.csv")
