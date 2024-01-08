import numpy as np
import sympy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import ticker, cm
from math import pi, exp, sin
from SA_func import f_c, light_speed, read_fft_data, code_V_convert, get_raw_data, chirp_rate, wl, d_tau

filename = "/Users/CHJ/Desktop/2023-12-04-12:41:32_vertical/fft_data"
fft_data = read_fft_data(filename)
data = code_V_convert(fft_data)
raw_data = get_raw_data(data)
index = [0, 500, 0, 120]
az_s_index = index[0]
az_e_index = index[1]
rg_s_index = index[2]
rg_e_index = index[3]
az_len = az_e_index - az_s_index
rg_len = rg_e_index - rg_s_index
all_font = 20


def DBF(raw_data):
    N = 8 # TX numbers * RX numbers = 8, number of arrays
    # center_freq = f_c # 24.15 GHz, the initial frequency of chirp of our FMCW radar
    d = 0.006 # 6 mmm, the distance between two adjacent RX antennas
    # c = light_speed
    phase_data = np.angle(raw_data[:,:,0:rg_e_index])
    phase_data = np.mod(phase_data, 2*pi) # make sure the phase is in the range of [0, 2*pi]
    # tau = phase_data / (2 * pi * center_freq) # time delay matrix
    𝜆 = wl # wavelength matrix
    N_col = np.reshape(np.arange(8),(1, 8)).T
    theta = np.arange(-90,91) # traverse all the angle from front direction
    theta = np.reshape(theta, (1,181))
    theta0 = np.angle(raw_data[:,:,0:rg_e_index]) # target beam angle
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
        # x = range(-90,91,1) # Plot the angle-amp figure//
        # y = result
        # plt.plot(x,y)
        # plt.xlabel('angle [˚]')
        # plt.ylabel('amp')#//
        angle.append(np.argmax(np.abs(result)) - 90)
        amp.append(max(np.abs(result)))

    # i = 0
    # for elem in 20*np.log10(np.abs(raw_data)): # elem is the 20log10(amp) of point
    #     rcs_image_36[int(angle[i]) + 90, i] = elem # put 20log10(amp) into the corresponding position(azim, angle)
    #     i += 1

    dx = 0.01 # 0.01s per point in azimuth direction, 10s/1000points
    dy = d_tau * light_speed / 2 / 2
    # plt.xticks(np.arange(0, az_e_index - az_s_index, step = x_step), np.round(np.arange(az_s_index * dx, az_e_index * dx, step = dx * x_step), 2), fontsize = all_font, rotation = 90)
    # plt.gca().invert_xaxis()
    # plt.yticks(np.arange(0, 180, step = y_step), np.round(np.arange(-90, 90, step = y_step), 2), fontsize = all_font, rotation = 0)
    # plt.gca().invert_yaxis()
    # plt.xlabel('azimuth [s]', fontsize = all_font)
    # plt.ylabel('angle [˚]', fontsize = all_font)
    # plt.tight_layout()
    # plt.title("RCS at 16.33m in range direction")
    # plt.show()

    azimuth_one_row = np.arange(0, 5, dx) + dx / 2 # (0, azimuth sampling seconds, dx), home
    azimuth = np.repeat(azimuth_one_row, rg_e_index)
    slant_range_one_col = np.arange(0, rg_e_index) * dy + dy / 2
    slant_range = np.tile(slant_range_one_col, (1, az_len))
    ground_range = np.cos((80+np.array(angle)) * pi / 180) * slant_range # elevation angle = 80 degree
    height = np.sin((80+np.array(angle)) * pi / 180) * slant_range

    # 1.1 根据各个点的值(v[])，设置点的颜色值，每个点的颜色使用一个rgb三维的元组表示，例如，若想让点显示为红色，则颜色值为(1.0,0,0)
    # 设置各个点的颜色
    # 每个点的颜色值按照colormap("seismic",100)进行设计，其中colormap类型为"seismic"，共分为100个级别(level)

    # 2.0 显示三维散点图
    # 新建一个figure()
    fig = plt.figure()
    # 在figure()中增加一个subplot，并且返回axes
    ax = fig.add_subplot(111,projection='3d')
    plt.set_cmap(plt.get_cmap("seismic", 100))
    plt.gca().invert_yaxis() # invert azimuth axis corresponding to the scan direction which is from right to left
    # amp[(amp < 8) | (amp > 20)] = None # set the amp of the point which is too low to None for eliminating noise
    amp_copy = np.array(amp)
    amp_copy[(amp_copy < 6.5)] = 0 # set the amp of the point which is too low to None for eliminating noise
    # mask = ~np.isnan(amp_copy) # mask the point which is None
    # azimuth = azimuth[mask]
    ground_range = np.reshape(ground_range, (az_len * rg_e_index))
    # ground_range = ground_range[mask]
    height = np.reshape(height, (az_len * rg_e_index))
    # height = height[mask]
    # amp_copy = amp_copy[mask]
    color = plt.get_cmap("jet")((amp_copy-np.nanmin(amp_copy)) / (np.nanmax(amp_copy)-np.nanmin(amp_copy)))
    im = ax.scatter(azimuth, ground_range, height, ((amp_copy-np.nanmin(amp_copy)) / (np.nanmax(amp_copy)-np.nanmin(amp_copy))), s=40,c=color,marker='.')
    # 2.1 增加侧边colorbar
    # 设置侧边colorbar，colorbar上显示的值使用lambda方程设置
    min_amp = min(amp_copy)
    max_amp = max(amp_copy)
    fig.colorbar(im, format=ticker.FuncFormatter(lambda x, pos: round(x*(max_amp-min_amp)+min_amp, 1))) # x:0~1, lambda:min_amp~max_amp
    ax.set_xlabel('azimuth [s]')
    ax.set_ylabel('ground range [m]')
    ax.set_zlabel('height [m]')
    plt.show()

DBF(raw_data)