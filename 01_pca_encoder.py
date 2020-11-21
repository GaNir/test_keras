# Create PCA encoder with orientation constraint
# Reference https://github.com/ageron/handson-ml
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from tensorflow import keras
from auxiliary import get_data


import numpy as np

import time
from datetime import datetime
import numpy.random as rnd
import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

assert tf.version.VERSION == '2.3.1'
assert tf.executing_eagerly() == True

# ---------- params -------------------------------------------
output_folder = 'outputs/'

ITERATIONS = 10
# -------------------------------------------------------------


now = datetime.now()
file_header = output_folder + now.strftime("%Y%m%d_%H%M%S_")


X = get_data()



# # ------- create dataset
# rnd.seed(4)
# m = 5000 # total samples
# test_samples = 1
# w1, w2 = 0.1, 0.3
# noise = 0.1
#
# angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
# data = np.empty((m, 3))
# data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
# data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
# data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)
#
# # Normalize
# scaler = StandardScaler()
# X_train = scaler.fit_transform(data[:m-test_samples])
# X_test = scaler.transform(data[m-test_samples:])


class my_model(keras.Model):
    def __init__(self, units=3, **kargs):
        super().__init__(**kargs)
        self.hidden_encode = keras.layers.Dense(2, input_shape=[3])
        self.hidden_decode = keras.layers.Dense(3)

    def call(self, inputs):
        encoded = self.hidden_encode(inputs)
        decoded = self.hidden_decode(encoded)
        # tmp = keras.Model(inputs=inputs, outputs=encoded, name="encoder")
        return decoded

    def get_encoder_model(self):
        return keras.models.Sequential([self.hidden_encode])

    # https://keras.io/guides/customizing_what_happens_in_fit/
    # https://stackoverflow.com/questions/53953099/what-is-the-purpose-of-the-tensorflow-gradient-tape
    # https://www.tensorflow.org/guide/eager

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        # print ('trainable_vars', trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

model = my_model(units = 3)
model.compile(loss = "mse")
history = model.fit(X, X, epochs = 1, verbose = 0)#, callbacks = [tensorboar_cb])
model_encoder = model.get_encoder_model()

X_align = [1,1,1]
y_align = [-0.37301758,  1.1074662]

X_align_train = np.array([X_align,X_align])
y_align_train = np.array([y_align,y_align])

model_encoder.compile(loss = "mse")

h1_list = []
h2_list = []


for i in range(ITERATIONS):
    print ('%d/%d      \r'%(i,ITERATIONS), end = '')
    history = model.fit(X, X, epochs = 5, verbose = 0)#, callbacks = [tensorboar_cb])
    align_history = model_encoder.fit(X_align_train, y_align_train, epochs = 15, verbose = 0)#, callbacks = [tensorboar_cb])
    h1_list.append(history.history['loss'])
    h2_list.append(align_history.history['loss'])

coding = model_encoder.predict(X)
align_coding_vect = model_encoder.predict(X_align_train)
align_coding = align_coding_vect[0]


# ---- plot results -----------------
plt.figure()
plt.plot(coding[:,0], coding[:,1],'.')
plt.plot(y_align[0], y_align[1], 'rx')
plt.plot(align_coding[0], align_coding[1], 'r.')

filename = file_header+'mapping.png'
print ('------------------------')
print (f'saving {filename}')
plt.savefig(filename)