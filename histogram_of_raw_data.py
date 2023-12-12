import numpy as np
import matplotlib.pyplot as plt

from SA_func import read_fft_data, code_V_convert, get_raw_data


filename = "/Users/CHJ/Desktop/2023-12-04-12:41:32_vertical/fft_data" # home
fft_data = read_fft_data(filename)
data = code_V_convert(fft_data)
raw_data = get_raw_data(data)

def his(raw_data):
    amp = np.abs(raw_data[0,:,:])
    amp = np.reshape(amp, (500*512))
    n, bins, patches = plt.hist(x=amp, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('amp')
    plt.ylabel('numbers')
    plt.show()

his(raw_data)