import numpy as np
import scipy.signal
from .ml_models import MLAudioProcessor

# Initialize ML processor as a singleton
_ml_processor = None

def get_ml_processor():
    global _ml_processor
    if _ml_processor is None:
        _ml_processor = MLAudioProcessor()
    return _ml_processor

def compute_stft(audio, rate, n_fft=1024, hop_length=512):
    """Compute the magnitude spectrogram using STFT"""
    f, t, Zxx = scipy.signal.stft(audio, fs=rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    return f, t, np.abs(Zxx)

def vad_energy(audio, rate, frame_duration_ms=30, threshold=1e-4, use_ml=True):
    """
    Voice Activity Detection using either ML-based or energy-based approach.
    
    Args:
        audio: Input audio signal
        rate: Sampling rate
        frame_duration_ms: Frame duration in milliseconds (for energy-based VAD)
        threshold: Energy threshold (for energy-based VAD) or VAD confidence (for ML-based VAD)
        use_ml: Whether to use ML-based VAD (True) or energy-based VAD (False)
    
    Returns:
        Processed audio with silent parts removed or zeroed out
    """
    if use_ml:
        try:
            ml_proc = get_ml_processor()
            vad_mask = ml_proc.detect_voice_activity(audio, rate, threshold)
            return audio * vad_mask
        except Exception as e:
            print(f"Warning: ML-based VAD failed ({str(e)}), falling back to energy-based VAD")
            return vad_energy(audio, rate, frame_duration_ms, threshold, use_ml=False)
    
    # Original energy-based VAD logic
    frame_length = int(rate * frame_duration_ms / 1000)
    if len(audio) < frame_length:
        return audio  # Short signal, skip VAD

    num_frames = len(audio) // frame_length
    speech_frames = []

    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length
        frame = audio[start:end]
        energy = np.sum(frame.astype(np.float32) ** 2) / len(frame)
        if energy > threshold:
            speech_frames.append(frame)
        else:
            speech_frames.append(np.zeros_like(frame))  # silence

    if not speech_frames:
        return np.zeros_like(audio)  # failsafe for silence

    return np.concatenate(speech_frames)
