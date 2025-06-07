import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(data, rate):
    times = np.arange(len(data)) / rate
    plt.figure(figsize=(10, 4))
    plt.plot(times, data)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
