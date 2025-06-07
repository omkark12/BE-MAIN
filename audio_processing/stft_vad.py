import numpy as np
import scipy.signal

def compute_stft(audio, rate, n_fft=1024, hop_length=512):
    """Compute the magnitude spectrogram using STFT"""
    f, t, Zxx = scipy.signal.stft(audio, fs=rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    return f, t, np.abs(Zxx)

def vad_energy(audio, rate, frame_duration_ms=30, threshold=1e-4):
    """Energy-based VAD with safer logic"""
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
        print(f"Frame {i}: Energy = {energy}")
        if energy > threshold:
            speech_frames.append(frame)
        else:
            speech_frames.append(np.zeros_like(frame))  # silence

    if not speech_frames:
        return np.zeros_like(audio)  # failsafe for silence

    return np.concatenate(speech_frames)
