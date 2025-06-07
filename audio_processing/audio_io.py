import pyaudio
import numpy as np
from audio_processing.noise_filter import fft_filter
from audio_processing.mvdr import mvdr_beamforming
from audio_processing.stft_vad import vad_energy, compute_stft
from audio_processing.ml_model import MLNoiseSuppressor
from audio_processing.mfcc_features import extract_mfcc


class AudioProcessor:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.running = False
        self.audio = pyaudio.PyAudio()
        self.ml_filter = MLNoiseSuppressor()

    def start_processing(self, intensity, mfcc_callback=None):
        self.running = True
        input_stream = self.audio.open(format=self.format, channels=self.channels,
                                       rate=self.rate, input=True, frames_per_buffer=self.chunk)
        output_stream = self.audio.open(format=self.format, channels=self.channels,
                                        rate=self.rate, output=True, frames_per_buffer=self.chunk)

        while self.running:
            data = input_stream.read(self.chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            voice_only = vad_energy(audio_data, self.rate)
            filtered_data = fft_filter(voice_only, self.rate, intensity)

            # ✅ Play filtered audio
            output_stream.write(filtered_data.tobytes())

            # ✅ Extract & send MFCC to GUI
            mfcc_features = extract_mfcc(filtered_data, self.rate)
            if mfcc_callback:
                mfcc_callback(mfcc_features)

            output_stream.write(filtered_data.tobytes())    
        # while self.running:
        #     data = input_stream.read(self.chunk, exception_on_overflow=False)
        #     audio_data = np.frombuffer(data, dtype=np.int16)

        #     voice_only = vad_energy(audio_data, self.rate)
        #     filtered_data = fft_filter(voice_only, self.rate, intensity)
        #     # Optional: Extract MFCCs
        #     mfcc_features = extract_mfcc(filtered_data, self.rate)
        #     print("MFCC shape:", mfcc_features.shape)  # (n_mfcc, time_frames)

        #     # # ML post-filter
        #     # if len(filtered_data) >= 1024:  # Ensure input size
        #     #     denoised = self.ml_filter.denoise(filtered_data[:1024])
        #     #     output_stream.write(denoised.tobytes())
        #     # else:
        #     #     output_stream.write(filtered_data.tobytes())
        #     mfcc_features = extract_mfcc(filtered_data, self.rate)
        #     if mfcc_callback:
        #         mfcc_callback(mfcc_features)

        #     output_stream.write(filtered_data.tobytes())


        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()

    def stop_processing(self):
        self.running = False
        self.audio.terminate()
