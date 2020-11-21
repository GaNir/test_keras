import numpy as np
from sklearn.preprocessing import StandardScaler

def get_data(samples_num = 5000):
    '''
        A simple 3D data set
    '''
    np.random.seed(4)
    w1, w2 = 0.1, 0.3
    noise = 0.1

    angles = np.random.rand(samples_num) * 3 * np.pi / 2 - 0.5
    data = np.empty((samples_num, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(samples_num) / 2
    data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(samples_num) / 2
    data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(samples_num)

    # Normalize
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data