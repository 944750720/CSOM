import cmaps
import numpy as np
import sympy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from cross_module import cross
from matplotlib import ticker, cm
from math import pi, exp, sin
from SA_func import f_c, light_speed, read_fft_data, code_V_convert, get_raw_data, chirp_rate, wl, d_tau
from CMOS import CMOS_process

filename = "/Users/CHJ/Desktop/send/10-23-17-41-58/fft_data"
fft_data = read_fft_data(filename)
data = code_V_convert(fft_data)
raw_data = get_raw_data(data)
index = [0, 1000, 0, 60]
az_s_index = index[0]
az_e_index = index[1]
rg_s_index = index[2]
rg_e_index = index[3]
az_len = az_e_index - az_s_index
# rg_s_index = index[2]
# rg_e_index = index[3]
# rg_len = rg_e_index - rg_s_index
all_font = 20

def DBF(raw_data, class_labels):
    N = 8 # TX numbers * RX numbers = 8, number of arrays
    # center_freq = f_c # 24.15 GHz, the initial frequency of chirp of our FMCW radar
    d = 0.006 # 5 mmm, the distance between two adjacent RX antennas
    # c = light_speed
    phase_data = np.angle(raw_data[:,:,0:60])
    phase_data = np.mod(phase_data, 2*pi) # make sure the phase is in the range of [0, 2*pi]
    # tau = phase_data / (2 * pi * center_freq) # time delay matrix
    𝜆 = wl # wavelength matrix
    N_col = np.reshape(np.arange(8),(1, 8)).T
    theta = np.arange(-90,91) # traverse all the angle from front direction
    theta = np.reshape(theta, (1,181))
    theta0 = np.angle(raw_data[:,:,0:60]) # target beam angle
    theta0 = np.reshape(theta0, (8,-1))
    A = - N_col * (1j * 2 * pi * d * np.sin(theta*pi/180) / 𝜆 )
    target_beam = np.exp( 1j * 2 * pi * d  * np.sin(theta0) / 𝜆 ).T
    steering_vector = np.exp(A)
    angle = []
    for col_tar in target_beam:
        result = np.array([])
        for col in steering_vector.T:
            result = np.append(result, np.dot(col, col_tar))
        angle.append(np.argmax(np.abs(result)) - 90)

    # plot 3d image
    dx = 0.01 # 0.01s per point in azimuth direction, 10s/1000points
    dy = d_tau * light_speed / 2 / 2 # 0.4537874120117187m, range resolution
    azimuth_one_row = np.arange(0, 10, dx) + dx / 2
    azimuth = np.repeat(azimuth_one_row, 60)
    slant_range_one_col = np.arange(0, 60) * dy + dy / 2
    slant_range = np.tile(slant_range_one_col, (1, az_len))
    ground_range = np.cos(np.array(angle) * pi / 180) * slant_range
    height = np.sin(np.array(angle) * pi / 180) * slant_range

    # 2.0 显示三维散点图
    # 新建一个figure()
    fig = plt.figure()
    # 在figure()中增加一个subplot，并且返回axes
    ax = fig.add_subplot(111,projection='3d')
    # 设置colormap，与上面提到的类似，使用"seismic"类型的colormap，共100个级别
    plt.set_cmap(plt.get_cmap(cmaps.amwg_blueyellowred, 100))
    plt.gca().invert_yaxis() # invert azimuth axis corresponding to the scan direction which is from right to left
    # 绘制三维散点，各个点颜色使用color列表中的值，形状为"."
    ground_range_start, ground_range_end = 15, 16
    azimuth, ground_range, height, class_labels = cross(azimuth, ground_range, height, class_labels, ground_range_start, ground_range_end) # plot the section from ground_range  to 16m
    # 1.1 根据各个点的值(v[])，设置点的颜色值，每个点的颜色使用一个rgb三维的元组表示，例如，若想让点显示为红色，则颜色值为(1.0,0,0)
    # 设置各个点的颜色
    # 每个点的颜色值按照colormap("seismic",100)进行设计，其中colormap类型为"seismic"，共分为100个级别(level)
    min_class_labels = min(class_labels)
    max_class_labels = max(class_labels)
    color = [plt.get_cmap("seismic", 100)(int(float(i-min_class_labels)/(max_class_labels-min_class_labels)*100)) for i in class_labels]
    im = ax.scatter(azimuth, ground_range, height, class_labels, s=100,c=color,marker='.')
    # 2.1 增加侧边colorbar
    # 设置侧边colorbar，colorbar上显示的值使用lambda方程设置
    fig.colorbar(im, format=ticker.FuncFormatter(lambda x,pos:int(x*(max_class_labels-min_class_labels)+min_class_labels)))
    ax.set_xlabel('azimuth [s]')
    ax.set_ylabel('ground range [m]')
    ax.set_zlabel('height [m]')
    plt.show()

class_labels = CMOS_process(raw_data, index, az_s_index, az_e_index, rg_s_index, rg_e_index)
DBF(raw_data,class_labels)