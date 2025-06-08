import numpy as np
from typing import Optional, Tuple, Dict
from .ml_models import MLAudioProcessor
import librosa
from scipy.ndimage import gaussian_filter1d

# Initialize ML processor as a singleton
_ml_processor = None

def get_ml_processor():
    global _ml_processor
    if _ml_processor is None:
        _ml_processor = MLAudioProcessor()
    return _ml_processor

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Convert audio data to float32 in range [-1, 1]"""
    if audio_data.dtype == np.int16:
        return audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.float32:
        return np.clip(audio_data, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported audio data type: {audio_data.dtype}")

def denormalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Convert float32 audio back to int16"""
    if audio_data.dtype == np.float32:
        # Clip to [-1, 1] range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        # Convert to int16
        return (audio_data * 32767).astype(np.int16)
    elif audio_data.dtype == np.int16:
        return audio_data
    else:
        raise ValueError(f"Unsupported audio data type: {audio_data.dtype}")

def fft_filter(data: np.ndarray, rate: int, intensity: float = 50) -> np.ndarray:
    """
    Legacy FFT-based noise filtering method.
    Kept for fallback and comparison purposes.
    """
    # Convert to float32 for processing
    float_data = normalize_audio(data)
    
    # Apply FFT
    fft_data = np.fft.fft(float_data)
    frequencies = np.fft.fftfreq(len(float_data), 1 / rate)

    # Zero out low-magnitude frequencies based on intensity
    magnitude_threshold = np.percentile(np.abs(fft_data), intensity)
    fft_data[np.abs(fft_data) < magnitude_threshold] = 0

    # Convert back to time domain
    filtered_data = np.fft.ifft(fft_data).real
    
    # Ensure the result is in [-1, 1] range
    filtered_data = np.clip(filtered_data, -1.0, 1.0)
    
    # Convert back to int16
    return denormalize_audio(filtered_data)

def analyze_noise(data: np.ndarray, rate: int) -> Dict[str, float]:
    """
    Analyze and classify the type of noise in the audio.
    
    Args:
        data: Input audio signal
        rate: Sampling rate
    
    Returns:
        Dictionary containing noise type and confidence score
    """
    try:
        ml_proc = get_ml_processor()
        # Convert to float32 for ML processing
        float_data = normalize_audio(data)
        return ml_proc.classify_noise(float_data, rate)
    except Exception as e:
        print(f"Warning: Noise classification failed: {str(e)}")
        return {'noise_type': 'unknown', 'confidence': 0.0}

def filter_noise(data: np.ndarray, rate: int, 
                use_ml: bool = True,
                intensity: float = 50) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Enhanced noise filtering with optional ML-based approach and noise analysis.
    
    Args:
        data: Input audio signal (int16 or float32)
        rate: Sampling rate
        use_ml: Whether to use ML-based denoising (True) or FFT-based filtering (False)
        intensity: Intensity of FFT filtering if ML-based approach fails
    
    Returns:
        Tuple of (filtered audio, noise analysis results)
    """
    noise_info = None
    
    try:
        # Convert to float32 for processing
        float_data = normalize_audio(data)
        
        # Analyze noise type
        try:
            ml_proc = get_ml_processor()
            noise_info = ml_proc.classify_noise(float_data, rate)
        except Exception as e:
            print(f"Warning: Noise classification failed: {str(e)}")
            noise_info = {'noise_type': 'unknown', 'confidence': 0.0}
        
        # Use robust noise filtering
        return filter_noise_robust(data, rate, noise_info['noise_type']), noise_info
            
    except Exception as e:
        print(f"Warning: Noise filtering failed: {str(e)}")
        return data, noise_info

def filter_noise_robust(data: np.ndarray, rate: int, noise_type: str = None) -> np.ndarray:
    """
    Robust noise filtering using spectral gating and frequency-selective processing
    """
    try:
        # Convert to float32
        float_data = normalize_audio(data)
        
        # FFT parameters
        n_fft = 2048
        hop_length = 512
        win_length = 2048
        
        # Compute STFT
        D = librosa.stft(float_data,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window='hann',
                        center=True)
        
        # Get magnitude and phase
        mag = np.abs(D)
        phase = np.angle(D)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
        
        # Define frequency ranges
        voice_ranges = [
            (85, 255),     # Male fundamental
            (165, 400),    # Female fundamental
            (400, 2000),   # Main formants
            (2000, 3500)   # Consonants
        ]
        
        # Create frequency mask
        freq_mask = np.ones(len(freqs))
        
        # Set different reduction levels for different frequency ranges
        for low, high in voice_ranges:
            mask = (freqs >= low) & (freqs <= high)
            if low == 400 and high == 2000:  # Formants
                freq_mask[mask] = 0.3  # Preserve more
            elif low >= 2000:  # Consonants
                freq_mask[mask] = 0.4
            else:  # Fundamentals
                freq_mask[mask] = 0.35
        
        # More aggressive reduction outside voice ranges
        freq_mask[freqs < 85] = 0.95    # Very low frequencies
        freq_mask[freqs > 3500] = 0.92  # High frequencies
        
        # Noise type specific parameters
        reduction_strength = {
            'background_noise': 0.95,
            'music': 0.92,
            'machine': 0.97,
            'white_noise': 0.97,
            'speech': 0.80,
            'other': 0.90
        }.get(noise_type, 0.90)
        
        # Estimate noise floor
        noise_floor = np.percentile(mag, 20, axis=1)
        
        # Apply spectral gating
        gain_mask = np.maximum(
            1.0 - reduction_strength * freq_mask[:, np.newaxis] * (
                noise_floor[:, np.newaxis] / (mag + 1e-10)
            ),
            0.02  # Minimum gain
        )
        
        # Smooth the mask
        gain_mask = gaussian_filter1d(gain_mask, sigma=1, axis=1)
        
        # Apply mask
        mag_clean = mag * gain_mask
        
        # Boost voice frequencies
        formant_mask = (freqs >= 400) & (freqs <= 2000)
        consonant_mask = (freqs >= 2000) & (freqs <= 3500)
        
        mag_clean[formant_mask] *= 1.2    # Boost formants
        mag_clean[consonant_mask] *= 1.15  # Boost consonants
        
        # Reconstruct signal
        D_clean = mag_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(D_clean,
                                  hop_length=hop_length,
                                  win_length=win_length,
                                  window='hann',
                                  center=True)
        
        # Ensure output length matches input
        if len(audio_clean) > len(float_data):
            audio_clean = audio_clean[:len(float_data)]
        elif len(audio_clean) < len(float_data):
            audio_clean = np.pad(audio_clean, (0, len(float_data) - len(audio_clean)))
        
        # Convert back to original format
        return denormalize_audio(audio_clean)
        
    except Exception as e:
        print(f"Robust noise filtering error: {str(e)}")
        return data
