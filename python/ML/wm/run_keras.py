import numpy as np

import pandas as pd
import keras
import tensorflow as tf
import keras.backend as K

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
Step 1:
Question: Can a neural network approximate the law of the wall?
Answer:   It requires some tuning, but seems to give reasonable results
Step 2:
Question: Can we customize the neural network so that tau.u_w is always positive?
Step 3:
Non-dimensionalize and check if it generalizes
"""

wm_data  = pd.read_csv("wm_data.csv")

x_smpls = pd.DataFrame(data=wm_data, columns=["h", "u_L", "u_w"]) 
y       = pd.DataFrame(data=wm_data, columns=["tau"]) 

# define the keras model
model = Sequential()
# Add layers
"""
Repeat a few times before finalizing network 
"""
model.add(Dense(15, input_dim=3, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense( 1, activation='tanh'))

#Examine the model
print(model.summary())


########################################
## Test a functionality
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
########################################

########################################
#First let us do a custom loss function
# Define custom loss
def custom_loss():

  def loss(y_true,y_pred):
      return K.mean( K.square(K.square( y_pred ) - y_true) )

  # Return a function
  return loss

## END FUNCTION 
########################################

# compile the keras model
#model.compile(loss=custom_loss(), optimizer='adam', metrics=[mean_pred])
model.compile(loss=custom_loss(), optimizer='adam')
# fit the keras model on the dataset
model.fit(x_smpls, y, validation_split=0.3, epochs=50, verbose=2)

weights = model.get_weights()

space   = Space([(10., 100.), (2., 4.), (-1., 1.)])
lhs     = Lhs(lhs_type="classic", criterion=None)
x_n     = lhs.generate(space.dimensions, 10)

x_tst   = pd.DataFrame(data=x_n, columns=["h", "u_L", "u_w"]) 
n_smpls = x_smpls.shape[0]

y_prd   = K.square( model.predict(x_tst) )

nu  = 0.001
for it, i in enumerate(x_n):
  h   = i[0]
  u_L = i[1]

  tau = getTau(nu, h, u_L) 

  tf.print(i, tau, [y_prd[it]])



