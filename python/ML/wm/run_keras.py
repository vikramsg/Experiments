# first neural network with keras tutorial
import numpy as np

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

from skopt.space import Space
from skopt.sampler import Lhs

from tauGrad import *

"""
Given tau_w(u_LES, h) through an implicit equation 
we want to convert it into tau_w(u_LES, u_w, h)
such that u_w \cdot tau_w > 0
to ensure kinetic energy stability
The first step is to check whether a neural network
can approximate the law of the wall
"""

wm_data  = pd.read_csv("wm_data.csv")

x_smpls = pd.DataFrame(data=wm_data, columns=["h", "u"]) 
y       = pd.DataFrame(data=wm_data, columns=["tau"]) 

# define the keras model
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense( 5, activation='relu'))
model.add(Dense(1, activation='tanh'))

#Examine the model
print(model.summary())

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam')
# fit the keras model on the dataset
model.fit(x_smpls, y, validation_split=0.2, epochs=25, verbose=2)

space   = Space([(2., 4.), (10., 100.)])
lhs     = Lhs(lhs_type="classic", criterion=None)
x_n     = lhs.generate(space.dimensions, 10)

x_tst   = pd.DataFrame(data=x_n, columns=["h", "u"]) 
n_smpls = x_smpls.shape[0]

y_prd   = model.predict(x_tst)

nu  = 0.001
for it, i in enumerate(x_n):
  u   = i[0]
  h   = i[1]

  tau = 10.*getTau(nu, h, u) 

  print(i, tau, y_prd[it])















