# Create PCA encoder with orientation constraint
# Reference https://github.com/ageron/handson-ml

#IPython.core.debugger.Pdb.set_trace()

# %env PYTHONHASHSEED=0
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from tensorflow import keras
from auxiliary import get_data
from sklearn.decomposition import PCA


import numpy as np

import time
from datetime import datetime
import numpy.random as rnd
import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

assert tf.version.VERSION == '2.3.1'
assert tf.executing_eagerly() == True

# np.random.seed(123)
# python_random.seed(123)
# tf.random.set_seed(1234)

# ---------- params -------------------------------------------
output_folder = 'output/'
ITERATIONS = 200
model_count = 5
refresh_epoch = 10
# -------------------------------------------------------------
now = datetime.now()
# save_folder = output_folder + now.strftime('Y%m%d_%H%M%S_pca_constraint/')
save_folder = output_folder + 'pca_constraint/'

X = get_data()
X = X/5
X_pca = PCA(n_components=2).fit_transform(X)


class my_model(keras.Model):
    def __init__(self, units=3, X_align=None, y_align=None, **kargs):
        super().__init__(**kargs)
        #         self.hidden_encode = keras.layers.Dense(2, input_shape=[units],activation=activation)
        #         self.hidden_decode = keras.layers.Dense(3,activation=activation)
        self.hidden_encode = keras.layers.Dense(2, input_shape=[units])
        self.hidden_decode = keras.layers.Dense(3)

        self.X_align = X_align
        self.y_align = y_align

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_base_tracker = keras.metrics.Mean(name="loss_base")
        self.loss_shift_tracker = keras.metrics.Mean(name="loss_shift")

    def call(self, inputs):
        encoded = self.hidden_encode(inputs)
        decoded = self.hidden_decode(encoded)
        return decoded

    def latent_dim(self, inputs):
        encoded = self.hidden_encode(inputs)
        return encoded

    def get_encoder_model(self):
        return keras.models.Sequential([self.hidden_encode])

    # https://keras.io/guides/customizing_what_happens_in_fit/
    # https://stackoverflow.com/questions/53953099/what-is-the-purpose-of-the-tensorflow-gradient-tape
    # https://www.tensorflow.org/guide/eager

    def train_step(self, data):
        x, y = data

        # from IPython.core.debugger import Tracer; Tracer()()

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            y_align_pred = self.latent_dim(np.array(self.X_align))
            shift_loss = tf.reduce_sum(tf.square(y_align_pred - np.array(self.y_align)))
            base_loss = tf.reduce_sum(tf.square(y - y_pred))
            var_loss = 1 / 100 * tf.math.reduce_std(y_pred)
            #loss = base_loss + 1 / x.shape[0] * shift_loss + var_loss
            loss = base_loss + shift_loss + var_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.loss_base_tracker.update_state(base_loss)
        self.loss_shift_tracker.update_state(shift_loss)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        dict_res = {m.name: m.result() for m in self.metrics}
        dict_res.update({"base_loss": self.loss_base_tracker.result(),
                         "shift_loss": self.loss_shift_tracker.result(),
                         "loss": self.loss_tracker.result()})
        return dict_res


X_align2 = np.array([[-2.10512441e-01, 1.46262716e-04, -2.68479696e-02], [0, 0, 0]], dtype='float32')
y_align2 = np.array([[0.12980571, -0.16003226], [0, 0]], dtype='float32')
# X_align2 = np.array([[0, 0, 0] ,[0, 0, 0]], dtype = 'float32')
# y_align2 = np.array([[0, 0] ,[0, 0]], dtype = 'float32')


# opt = keras.optimizers.SGD(learning_rate=0.1)
opt = keras.optimizers.SGD(lr=0.01, nesterov=True, learning_rate=0.05)


#  Reset
frame_counter = 0
model_vect = []
for m in range(model_count):
    model = my_model(units=3, X_align=X_align2, y_align=y_align2)
    model.compile(loss = 'mse', optimizer = opt)
    model_vect.append(model)

if not os.path.exists(save_folder):
    print ('Creating',save_folder)
    os.makedirs(save_folder)


plt.figure(0)
nrows = model_count
ncols = 2

h_vect = np.ones([model_count, ITERATIONS*refresh_epoch]) * np.nan
sl_vect = np.ones([model_count, ITERATIONS*refresh_epoch]) * np.nan

for i in range(ITERATIONS):
    print(f'{i}/{ITERATIONS}             \r', end='')
    fig = plt.figure(0)
    fig.clf()
    ax = fig.subplots(nrows=nrows, ncols=ncols)
    # display.clear_output(wait=True)
    fig.suptitle('Epoch %d' % (frame_counter * refresh_epoch))
    for j, model in enumerate(model_vect):
        history = model.fit(X, X, epochs=refresh_epoch, verbose=0, batch_size=10)  # , callbacks = [tensorboar_cb])
        h_vect[j, i * refresh_epoch:(i + 1) * refresh_epoch] = history.history['loss']
        sl_vect[j, i * refresh_epoch:(i + 1) * refresh_epoch] = history.history['shift_loss']


        ax[j][0].plot(np.log(h_vect[j]), 'C0')
        ax[j][0].plot(np.log(sl_vect[j]), 'C1')

        # model.summary()
        # print(f'Model {m_count + 1}/{MODEL_NUM} loss')

        # --------------------------------------
        model_encoder = model.get_encoder_model()
        model_encoder.compile(loss="mse")
        coding = model_encoder.predict(X)

        align_coding_vect = model_encoder.predict(X_align2)

        ax[j][1].plot(coding[::10, 0], coding[::10, 1], '.', alpha=0.1)
        ax[j][1].plot(y_align2[:, 0], y_align2[:, 1], 'rx')
        ax[j][1].plot(align_coding_vect[:, 0], align_coding_vect[:, 1], 'r.')

        # ------

    frame_counter = frame_counter + 1
    plt.savefig(save_folder + 'im%05d' % frame_counter)
    plt.show(block=False)
    # display.display(plt.gcf())