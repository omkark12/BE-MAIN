import numpy as np

def fft_filter(data, rate, intensity=50):
    fft_data = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), 1 / rate)

    # Zero out low-magnitude frequencies based on intensity
    magnitude_threshold = np.percentile(np.abs(fft_data), intensity)
    fft_data[np.abs(fft_data) < magnitude_threshold] = 0

    filtered_data = np.fft.ifft(fft_data).real
    return filtered_data.astype(np.int16)
