from keras.losses import binary_crossentropy,logcosh
import numpy as np

from keras import backend as K
y_true = K.variable(np.array([[1,2], [1,2], [1,2], [1,2]]))
y_pred = K.variable(np.array([[3,3], [4,6], [3,3], [3,3]]))
error = K.eval(logcosh(y_true, y_pred))

print(error)