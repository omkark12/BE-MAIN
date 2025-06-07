import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def plot_waveform(data, rate):
    times = np.arange(len(data)) / rate
    plt.figure(figsize=(10, 4))
    plt.plot(times, data)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

class MFFCCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e272e')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_mfcc(self, mfcc, sr):
        self.ax.clear()
        self.ax.set_facecolor('#1e272e')
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=self.ax)
        self.ax.set_title("MFCC", color='white')
        self.fig.tight_layout()
        self.draw()
