from math import pi, exp, sin
from SA_func import f_c, light_speed, read_fft_data, code_V_convert, get_raw_data, chirp_rate, wl, d_tau

import numpy as np
import sympy as sp
import seaborn as sns
import matplotlib.pyplot as plt
filename = "/Users/CHJ/Desktop/send/10-23-17-41-58/fft_data"
fft_data = read_fft_data(filename)
data = code_V_convert(fft_data)
raw_data = get_raw_data(data)
index = [0, 1000, 0, 60]
az_s_index = index[0]
az_e_index = index[1]
az_len = az_e_index - az_s_index
# rg_s_index = index[2]
# rg_e_index = index[3]
# rg_len = rg_e_index - rg_s_index
all_font = 20


def DBF(raw_data):
    N = 8 # TX numbers * RX numbers = 8, number of arrays
    # center_freq = f_c # 24.15 GHz, the initial frequency of chirp of our FMCW radar
    d = 0.006 # 5 mmm, the distance between two adjacent RX antennas
    # c = light_speed
    phase_data = np.angle(raw_data)
    phase_data = np.mod(phase_data, 2*pi) # make sure the phase is in the range of [0, 2*pi]
    # tau = phase_data / (2 * pi * center_freq) # time delay matrix
    𝜆 = wl # wavelength matrix
    N_col = np.reshape(np.arange(8),(1, 8)).T
    theta = np.arange(-90,91) # traverse all the angle from front direction
    theta = np.reshape(theta, (1,181))
    theta0 = np.angle(raw_data[:,:,36]) # target beam angle
    theta0 = np.reshape(theta0, (8,-1))
    A = -N_col * (1j * 2 * pi * d * np.sin(theta*pi/180) / 𝜆 )
    target_beam = np.exp( 1j * 2 * pi * d  * np.sin(theta0) / 𝜆 ).T
    steering_vector = np.exp(A)
    angle = []
    for col_tar in target_beam:
        result = np.array([])
        for col in steering_vector.T:
            result = np.append(result, np.dot(col, col_tar))
            # np.append(result, np.dot(col, col_tar))
        # result = np.reshape(result, (181))
        angle.append(np.argmax(np.abs(result)) - 90)

    rcs_image_36 = np.zeros((181, 1000))
    i = 0
    for elem in 20*np.log10(np.abs(raw_data[0,:,36])):
        rcs_image_36[int(angle[i]) + 90, i] = elem
        i += 1

    sns.heatmap(rcs_image_36, cmap = "jet", vmin = -30, vmax = 30)
    dx = 0.01
    x_step = int(az_len / 20)
    y_step = 10
    plt.xticks(np.arange(0, az_e_index - az_s_index, step = x_step), np.round(np.arange(az_s_index * dx, az_e_index * dx, step = dx * x_step), 2), fontsize = all_font, rotation = 90)
    plt.gca().invert_xaxis()
    plt.yticks(np.arange(0, 180, step = y_step), np.round(np.arange(-90, 90, step = y_step), 2), fontsize = all_font, rotation = 0)
    plt.gca().invert_yaxis()
    plt.xlabel('azimuth [s]', fontsize = all_font)
    plt.ylabel('angle [˚]', fontsize = all_font)
    plt.tight_layout()
    plt.title("RCS at 16.33m in range direction")
    plt.show()

DBF(raw_data)