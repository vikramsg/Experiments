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

n_samples = 2000 
space = Space([(10., 100.), (2., 4.), (-1., 1.)])

lhs = Lhs(lhs_type="classic", criterion=None)
x   = lhs.generate(space.dimensions, n_samples)

x_smpls = pd.DataFrame(data=x, columns=["h", "u_L", "u_w"]) 
n_smpls = x_smpls.shape[0]

tau     = np.zeros(n_smpls)

nu  = 0.001
for it, i in enumerate(x):
  h   = i[0]
  u_L = i[1]
  u_w = i[2]

  tau[it] = getTau(nu, h, u_L) 

  print(i, tau[it])

y  = pd.DataFrame(data=tau, columns=["tau"]) 

wm_data = pd.concat([x_smpls, y], axis=1)
wm_data.to_csv("wm_data.csv")
