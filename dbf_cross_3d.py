import numpy as np
import sympy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from cross_module import cross
from matplotlib import ticker, cm
from math import pi, exp, sin
from SA_func import f_c, light_speed, read_fft_data, code_V_convert, get_raw_data, chirp_rate, wl, d_tau

filename = "/Users/CHJ/Desktop/send/10-23-17-41-58/fft_data" # school
# filename = "/Users/CHJ/Desktop/2023-12-04-12:41:32_vertical/fft_data" # home
fft_data = read_fft_data(filename)
data = code_V_convert(fft_data)
raw_data = get_raw_data(data)
# index = [0, 500, 0, 120] # home
index = [0, 1000, 0, 60] # school
az_s_index = index[0]
az_e_index = index[1]
rg_s_index = index[2]
rg_e_index = index[3]
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
    # phase_data = np.angle(raw_data[:,:,0:120]) # home
    phase_data = np.angle(raw_data[:,:,0:60]) # school
    phase_data = np.mod(phase_data, 2*pi) # make sure the phase is in the range of [0, 2*pi]
    # tau = phase_data / (2 * pi * center_freq) # time delay matrix
    𝜆 = wl # wavelength matrix
    N_col = np.reshape(np.arange(8),(1, 8)).T
    theta = np.arange(-90,91) # traverse all the angle from front direction
    theta = np.reshape(theta, (1,181))
    # theta0 = np.angle(raw_data[:,:,0:120]) # target beam angle, home
    theta0 = np.angle(raw_data[:,:,0:60]) # target beam angle, school
    theta0 = np.reshape(theta0, (8,-1))
    A = - N_col * (1j * 2 * pi * d * np.sin(theta*pi/180) / 𝜆 )
    steering_vector = np.exp( 1j * 2 * pi * d  * np.sin(theta0) / 𝜆 ).T
    weight_vector = np.exp(A)
    angle = []
    dbf_amp = []
    for col_tar in steering_vector:
        result = np.array([])
        for col in weight_vector.T:
            result = np.append(result, np.dot(col, col_tar))
        # result = np.reshape(result, (181))
        angle.append(np.argmax(np.abs(result)) - 90)
        dbf_amp.append(max(np.abs(result)))

    dx = 0.01 # 0.01s per point in azimuth direction, 10s/1000points
    dy = d_tau * light_speed / 2 / 2
    # azimuth_one_row = np.arange(0, 5, dx) + dx / 2 # (0, azimuth sampling seconds, dx), home
    azimuth_one_row = np.arange(0, 10, dx) + dx / 2 # (0, azimuth sampling seconds, dx), school
    # azimuth = np.repeat(azimuth_one_row, 120) # home
    azimuth = np.repeat(azimuth_one_row, 60) # school
    # slant_range_one_col = np.arange(0, 120) * dy + dy / 2 # home
    slant_range_one_col = np.arange(0, 60) * dy + dy / 2 # school
    slant_range = np.tile(slant_range_one_col, (1, az_len))
    ground_range = np.cos(np.array(angle) * pi / 180) * slant_range
    height = np.sin(np.array(angle) * pi / 180) * slant_range
    # amp = np.reshape(20*np.log10(np.abs(raw_data[0,:,0:60])), (az_len * rg_e_index))

    # plot 3d image
    fig = plt.figure()
    # Add a subplot to the figure() and return the axes
    ax = fig.add_subplot(111,projection='3d')
    # Set the colormap, similar to the one mentioned above, using the "seismic" type colormap with 100 levels
    # plt.set_cmap(plt.get_cmap("jet", 100))
    plt.gca().invert_yaxis() # invert azimuth axis corresponding to the scan direction which is from right to left
    # Plot 3D scatter plot, each point's color is determined by the values in the color list, and the shape is "."
    ground_range_start, ground_range_end = 30, 50 # from m to 50m
    # azimuth, ground_range, height, amp = cross(azimuth, ground_range, height, amp, ground_range_start, ground_range_end) # plot the section from ground_range to 16m (amp of TX2-RX1)
    azimuth, ground_range, height, dbf_amp = cross(azimuth, ground_range, height, dbf_amp, ground_range_start, ground_range_end) # plot the section from ground_range to 16m (dbf_amp)
    # 1.1 Set the color of each point based on its value (v[]), where the color of each point is represented by an RGB tuple. For example, if you want the point to be red, the color value should be (1.0, 0, 0).
    # Set the color for each point
    # The color value for each point is designed based on the colormap "jet" with 100 levels
    min_amp = min(np.reshape(dbf_amp, len(dbf_amp)))
    max_amp = max(np.reshape(dbf_amp, len(dbf_amp)))
    color = [plt.get_cmap("jet", 100)(int(float(i-min_amp)/(max_amp-min_amp)*100)) for i in dbf_amp]
    im = ax.scatter(azimuth, ground_range, height, dbf_amp, s=5,c=color,marker='.')
    # Set the colorbar on the side, format the displayed values using a lambda function
    fig.colorbar(im, format=ticker.FuncFormatter(lambda x,pos:int(x*(max_amp-min_amp)+min_amp)))
    ax.set_xlabel('azimuth [s]')
    ax.set_ylabel('ground range [m]')
    ax.set_zlabel('height [m]')
    plt.show()

DBF(raw_data)