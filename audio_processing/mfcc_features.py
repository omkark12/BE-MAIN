import numpy as np
import librosa

def extract_mfcc(audio_data, sample_rate=44100, n_mfcc=13):
    """
    Extract MFCCs from raw audio data.
    """
    # Convert int16 to float32
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)

    return mfccs  # shape: (n_mfcc, time_frames)
