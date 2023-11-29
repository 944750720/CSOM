import sys
sys.path.append('/Users/CHJ/文稿/weekly_report/CSOM/minisom')
sys.path.append(r'/Users/CHJ/文稿/无人机sar/yamakawa/pi_data/SAR_program')

import cmaps
import cmath
import CSOM_ring
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import SA_func as sa
import seaborn as sns
import sympy as sp

from skimage import io

## defining the parameters
ch = 8 # channel
ad_samp_point = 512  # sampling points in the range direction
az_n = 1000 # pixels in the azimuth direction
az_dt =0.01
light_speed = sa.light_speed
df = sa.df
dr = sa.dr
ad_samp_point = sa.ad_samp_point
chirp_rate = sa.chirp_rate
d_tau = df / chirp_rate
dir_name = "/Users/CHJ/Desktop/send/10-23-17-41-58/"
add_name = ""
filename = "fft_data"

# index = [0, 1000, 0, 60]
# az_s_index = index[0]
# az_e_index = index[1]
# az_len = az_e_index - az_s_index
# rg_s_index = index[2]
# rg_e_index = index[3]
# rg_len = rg_e_index - rg_s_index

def CMOS_process(raw_data, index, az_s_index, az_e_index, rg_s_index, rg_e_index):
    # (60, 1000, 8)極座標複素数配列に変換
    complex_val = []
    for i in range(8):
        data = raw_data[i] #raw_data (8,1000,60)
        # amp_data = 20 * np.log10(np.abs(data[az_s_index:az_e_index, rg_s_index:rg_e_index]))
        amp_data = np.abs(data[az_s_index:az_e_index, rg_s_index:rg_e_index])
        amp_data = np.reshape(amp_data, (amp_data.shape[0], amp_data.shape[1], 1))
        phase_data = np.angle(data[az_s_index:az_e_index, rg_s_index:rg_e_index])
        phase_data = np.reshape(phase_data, (phase_data.shape[0], phase_data.shape[1], 1))
        # Make (500, 50, 8)
        x = 0
        complex_val_i = np.zeros((amp_data.shape[0], amp_data.shape[1], 1),dtype=complex)
        for amp_data_i, phase_data_i in zip(amp_data, phase_data):
            y = 0
            for amp_data_j, phase_data_j in zip(amp_data_i, phase_data_i):
                complex_val_i[x,y,0] = amp_data_j*cmath.exp(1j*phase_data_j)
                y += 1
            x += 1
        complex_val.append(complex_val_i)
    complex_val = np.stack(complex_val, axis=-1)
    complex_val_8_column = np.reshape(complex_val, (-1, 8))

    # Reshaping the pixels matrix
    # pixels = np.reshape(img, (img.shape[0]*img.shape[1], 1))
    pixels = complex_val_8_column
    # SOM initialization and training
    print('training...')
    som = CSOM_ring.MiniSom_ring(3, 8, sigma=0.,
                learning_rate=0.2, neighborhood_function='bubble', activation_distance='hermitian_product')
    # som.random_weights_init(pixels)
    starting_weights = som.get_weights().copy()  # Saving the starting weights
    som.train_ring(pixels,480000, random_order=True, verbose=False, use_epochs=False)

    print('quantization...')
    qnt = som.quantization(pixels)  # Quantize each pixels of the image

    print('building new image...')
    clustered = np.zeros((*amp_data.shape, 8), dtype=complex)
    for i, q in enumerate(qnt):  # Place the quantized values into a new image, i=index, q=quantized value
        clustered[np.unravel_index(i, shape=(amp_data.shape[0], amp_data.shape[1]))] = q
    labelled = np.zeros((*amp_data.shape, 1), dtype=np.float64)
    flattened = clustered.reshape(-1, clustered.shape[-1])

    # Initialize class labels and current label
    class_labels = np.zeros((flattened.shape[0],), dtype=int)
    current_label = 1
    # Traverse the flattened array
    for i, vector in enumerate(flattened):
        # If the vector is already labeled, skip it
        if class_labels[i] != 0:
            continue
        # Label the vector if it is not labeled before
        class_labels[i] = current_label
        # Search for the same vector in the remaining vectors, and label them
        for j in range(i+1, flattened.shape[0]):
            if np.array_equal(vector, flattened[j]):
                class_labels[j] = current_label
        current_label += 1
    # Print the class labels and the corresponding vectors
    for i in range(1, current_label+1):
        print("Class", i, ":")
        for j in range(len(class_labels)):
            if class_labels[j] == i:
                print([(cmath.polar(cur)[0], cmath.polar(cur)[1]/math.pi/sp.pi) for cur in flattened[j]])
                break
    return class_labels