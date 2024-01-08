import numpy as np
import matplotlib.pyplot as plt

from math import pi, exp, sin
from SA_func import f_c, light_speed, read_fft_data, code_V_convert, get_raw_data, chirp_rate, wl, d_tau


filename = "/Users/CHJ/Desktop/2023-12-04-12:41:32_vertical/fft_data" # home
# filename = "/Users/CHJ/Desktop/send/10-23-17-41-58/fft_data" # school
fft_data = read_fft_data(filename)
data = code_V_convert(fft_data)
raw_data = get_raw_data(data)

def his(raw_data):
    d = 0.006 # 5 mmm, the distance between two adjacent RX antennas
    𝜆 = wl # wavelength matrix
    # phase_data = np.angle(raw_data[:,:,0:60]) # school
    phase_data = np.angle(raw_data[:,:,0:120]) # home
    phase_data = np.mod(phase_data, 2*pi) # make sure the phase is in the range of [0, 2*pi]
    # tau = phase_data / (2 * pi * center_freq) # time delay matrix
    N_col = np.reshape(np.arange(8),(1, 8)).T
    theta = np.arange(-90,91) # traverse all the angle from front direction
    theta = np.reshape(theta, (1,181))
    # theta0 = np.angle(raw_data[:,:,0:60]) # target beam angle, school
    theta0 = np.angle(raw_data[:,:,0:120]) # target beam angle, home
    theta0 = np.reshape(theta0, (8,-1))
    A = - N_col * (1j * 2 * pi * d * np.sin(theta*pi/180) / 𝜆 )
    steering_vector = np.exp( 1j * 2 * pi * d  * np.sin(theta0) / 𝜆 ).T
    weight_vector = np.exp(A)
    angle = []
    amp = []
    for col_tar in steering_vector:
        result = np.array([])
        for col in weight_vector.T:
            result = np.append(result, np.dot(col, col_tar))
        result = np.reshape(result, (181))
        angle.append(np.argmax(np.abs(result)) - 90)
        amp.append(max(np.abs(result)))
    n, bins, patches = plt.hist(x=amp, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Histogram of data of channel TX2-RX1 (after fft, voltage convert and dbf)')
    plt.xlabel('amp')
    plt.ylabel('numbers')
    plt.show()

his(raw_data)