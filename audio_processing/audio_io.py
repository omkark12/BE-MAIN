import pyaudio
import numpy as np
from audio_processing.noise_filter import fft_filter
from audio_processing.mvdr import mvdr_beamforming

class AudioProcessor:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.running = False
        self.audio = pyaudio.PyAudio()

    def start_processing(self, intensity):
        self.running = True
        input_stream = self.audio.open(format=self.format, channels=self.channels,
                                       rate=self.rate, input=True, frames_per_buffer=self.chunk)
        output_stream = self.audio.open(format=self.format, channels=self.channels,
                                        rate=self.rate, output=True, frames_per_buffer=self.chunk)

        while self.running:
            data = input_stream.read(self.chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Apply noise cancellation
            filtered_data = fft_filter(audio_data, self.rate, intensity)
            output_stream.write(filtered_data.tobytes())

        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()

    def stop_processing(self):
        self.running = False
        self.audio.terminate()
