import numpy as np

def cross(azimuth, ground_range, height, amp, ground_range_star, ground_range_end):
    indices = np.where((ground_range >= ground_range_star) & (ground_range <= ground_range_end))
    azimuth = azimuth[indices[1]]
    ground_range = ground_range[indices]
    height = height[indices]
    amp = np.array(amp)
    amp = amp[indices[1]]
    return azimuth, ground_range, height, amp