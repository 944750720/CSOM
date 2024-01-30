from math import exp, pi, sin

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sympy as sp
from matplotlib import cm, ticker
from scipy.signal import find_peaks

from SA_func import (chirp_rate, code_V_convert, d_tau, f_c, get_raw_data,
                     light_speed, read_fft_data, wl)

filename = "/Users/CHJ/Desktop/2024-01-16_transformer/fft_data"
fft_data = read_fft_data(filename)
data = code_V_convert(fft_data)
raw_data = get_raw_data(data)
index = [0, 1000, 0, 12]
az_s_index = index[0]
az_e_index = index[1]
rg_s_index = index[2]
rg_e_index = index[3]
az_len = az_e_index - az_s_index
rg_len = rg_e_index - rg_s_index
all_font = 20


def DBF(raw_data):
    N = 8  # TX numbers * RX numbers = 8, number of arrays
    # center_freq = f_c # 24.15 GHz, the initial frequency of chirp of our FMCW radar
    d = 0.006  # 6 mmm, the distance between two adjacent RX antennas
    # c = light_speed
    phase_data = np.angle(raw_data[:, :, 0:rg_e_index])
    phase_data = np.mod(phase_data, 2 *
                        pi)  # make sure the phase is in the range of [0, 2*pi]
    # tau = phase_data / (2 * pi * center_freq) # time delay matrix
    𝜆 = wl  # wavelength matrix
    N_col = np.reshape(np.arange(8), (1, 8)).T
    angle_step = 0.5
    theta = np.arange(
        -90, 90 + angle_step,
        angle_step)  # traverse all the angle from front direction
    theta = np.reshape(theta, (1, -1))
    steering_vector = np.reshape(raw_data[:, :, 0:rg_e_index], (8, -1)).T
    A = -N_col * (1j * 2 * pi * d * np.sin(theta * pi / 180) / 𝜆
                  )  # matrix shape: 8*(180/angle_step)
    weight_vector = np.exp(A)
    angle, amp, phase = [], [], []
    for elem in steering_vector:
        result = np.array([])
        for col in weight_vector.T:
            result = np.append(result, np.dot(col, elem))
        result = np.reshape(result, (len(theta[0])))
        # x = np.arange(-90, 90+angle_step, angle_step) # Plot the angle-amp figure//
        # y = 20* np.log10(np.abs(result))
        # plt.plot(x,y)
        # plt.xlabel('angle [˚]')
        # plt.ylabel('amp[dB]')
        peak_amp = 20 * np.log10(np.max(np.abs(result)))  # 0~180
        peak_array = np.concatenate(
            [
                min(20 * np.log10(np.abs(result))),
                20 * np.log10(np.abs(result)),
                max(20 * np.log10(np.abs(result)))
            ],
            axis=None
        )  # add two points at begin and end to find peaks at head and tail
        peaks_x_coordinate, _ = find_peaks(peak_array,
                                           distance=16 / angle_step,
                                           height=peak_amp - 3)
        peaks_x_coordinate = peaks_x_coordinate - 1
        points_indecies_around_peak = np.array([])
        for elem in peaks_x_coordinate:
            points_indecies_around_peak = np.append(
                points_indecies_around_peak,
                np.arange(elem - 5, elem + 6).astype(float))
        points_indecies_around_peak[(points_indecies_around_peak < 0) | (
            points_indecies_around_peak >= len(theta[0])
        )] = None  # limit the points_indecies_around_peak in the range of 0~(180/angular_step)
        mask = ~np.isnan(points_indecies_around_peak)
        points_indecies_around_peak = points_indecies_around_peak[mask]
        angle_of_points_around_peak = points_indecies_around_peak * angle_step - 90
        amp_of_points_around_peak = 20 * np.log10(
            (np.abs(result)[points_indecies_around_peak.astype(int)]))
        phase_of_points_around_peak = np.angle(result)[
            points_indecies_around_peak.astype(int)]
        phase_of_points_around_peak = np.clip(phase_of_points_around_peak,
                                              -np.pi, np.pi)
        # plt.plot(angle_of_points_around_peak, amp_of_points_around_peak, "x")#//
        angle.append(angle_of_points_around_peak)
        amp.append(amp_of_points_around_peak)
        phase.append(phase_of_points_around_peak)

    dx = 0.01  # 0.01s per point in azimuth direction, 10s/1000points
    dy = d_tau * light_speed / 2 / 2

    start_azimuth = dx / 2  # start from the center of first sampled point
    start_slant_range = dy / 2
    i = 0
    azimuth, slant_range = [], []
    cur_azimuth = start_azimuth
    for array in angle:
        if i in range(0, len(angle), rg_len):
            cur_azimuth += dx
            cur_slant_range = start_slant_range
        points = len(array)
        azimuth.append(np.repeat(cur_azimuth, points))
        slant_range.append(np.repeat(cur_slant_range, points))
        cur_slant_range += dy
        i += 1
    azimuth_flatten = np.array([])
    for elem in azimuth:
        azimuth_flatten = np.append(azimuth_flatten, elem)
    slant_range_flatten = np.array([])
    for elem in slant_range:
        slant_range_flatten = np.append(slant_range_flatten, elem)
    angle_flatten = np.array([])
    for elem in angle:
        angle_flatten = np.append(angle_flatten, elem)
    amp_flatten = np.array([], dtype=object)
    for elem in amp:
        amp_flatten = np.append(amp_flatten, elem)
    phase_flatten = np.array([], dtype=object)
    for elem in phase:
        phase_flatten = np.append(phase_flatten, elem)
    ground_range = np.cos(
        (np.array(angle_flatten).flatten()) * pi / 180) * slant_range_flatten
    height = np.sin(
        (np.array(angle_flatten).flatten()) * pi / 180) * slant_range_flatten

    # Set the color values for each point based on their values (v[]).
    # Each point's color is represented by an RGB tuple. For example, if you want the point to be displayed in red, the color value would be (1.0, 0, 0).
    # Set the colors for each point.
    # The color values for each point are designed using the colormap "jet" with 100 levels.

    # Display the 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.set_cmap(plt.get_cmap("jet", 100))
    plt.gca().invert_yaxis(
    )  # invert azimuth axis corresponding to the scan direction which is from right to left
    amp_copy = amp_flatten.copy()
    amp_copy[(amp_copy < 1) | (ground_range < 3.8) | (
        ground_range > 5.5
    )] = None  # set the amp of the point which is too low to None for eliminating noise
    amp_copy = amp_copy.astype(float)
    phase_copy = phase_flatten.copy()
    mask = ~np.isnan(amp_copy)  # mask the point which is None
    azimuth_flatten = azimuth_flatten[mask]
    # ground_range = np.reshape(ground_range, (az_len * rg_e_index))
    ground_range = ground_range[mask]
    # height = np.reshape(height, (az_len * rg_e_index))
    height = height[mask]
    amp_copy = amp_copy[mask]
    phase_copy = phase_copy[mask]
    phase_copy = phase_copy.astype(float)
    color = plt.get_cmap("jet")(
        (phase_copy - np.nanmin(phase_copy)) /
        (np.nanmax(phase_copy) - np.nanmin(phase_copy)))
    im = ax.scatter(azimuth_flatten,
                    ground_range,
                    height,
                    phase_copy,
                    s=50,
                    c=color,
                    marker='.')
    min_amp = min(amp_copy)
    max_amp = max(amp_copy)
    min_phase = min(phase_copy)
    max_phase = max(phase_copy)
    fig.colorbar(im,
                 format=ticker.FuncFormatter(lambda x, pos: round(
                     x * (max_phase - min_phase) + min_phase, 1))).set_label(
                         'phase [rad]',
                         loc='top',
                         rotation=0,
                         fontsize=all_font,
                         labelpad=140)  # x:0~1, lambda:min_amp~max_amp
    ax.set_xlabel('azimuth [s]')
    ax.set_ylabel('ground range [m]')
    ax.set_zlabel('height [m]')
    plt.title('phase of transformer', fontsize=all_font)
    plt.show()


DBF(raw_data)