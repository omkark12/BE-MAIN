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
    Robust noise filtering with powerful voice enhancement and amplification
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
        
        # Define critical voice frequency ranges with importance and boost factors
        voice_ranges = [
            (80, 260, 0.15, 1.8),     # Male fundamental (boost 80%)
            (160, 420, 0.15, 1.8),    # Female fundamental (boost 80%)
            (300, 3000, 0.50, 2.0),   # Main speech band (boost 100%)
            (3000, 4000, 0.20, 1.6)   # Consonants and clarity (boost 60%)
        ]
        
        # Create frequency mask with voice preservation
        freq_mask = np.ones(len(freqs))
        
        # Create voice enhancement mask
        voice_enhance_mask = np.ones(len(freqs))
        
        # Set different reduction levels and enhancement for different frequency ranges
        for low, high, importance, boost in voice_ranges:
            mask = (freqs >= low) & (freqs <= high)
            # Lower values mean more preservation
            freq_mask[mask] = 0.10 + (1 - importance)  # Even more preservation
            voice_enhance_mask[mask] = boost  # Apply frequency-specific boost
        
        # More selective reduction outside voice ranges
        freq_mask[freqs < 80] = 0.98     # Very low frequencies
        freq_mask[freqs > 4000] = 0.95   # High frequencies
        
        # Noise type specific parameters - even more conservative for voice
        reduction_strength = {
            'background_noise': 0.82,  # Less aggressive
            'music': 0.85,            # Less aggressive
            'machine': 0.88,          # Less aggressive
            'white_noise': 0.90,
            'speech': 0.45,           # Even gentler on speech
            'other': 0.75
        }.get(noise_type, 0.75)
        
        # Estimate noise floor - more conservative
        noise_floor = np.percentile(mag, 10, axis=1)  # Lower percentile
        
        # Apply spectral gating with voice preservation
        gain_mask = np.maximum(
            1.0 - reduction_strength * freq_mask[:, np.newaxis] * (
                noise_floor[:, np.newaxis] / (mag + 1e-10)
            ),
            0.12  # Even higher minimum gain
        )
        
        # Smoother mask transitions
        gain_mask = gaussian_filter1d(gain_mask, sigma=1.2, axis=1)
        
        # Apply noise reduction mask
        mag_clean = mag * gain_mask
        
        # Apply voice enhancement
        voice_enhance_mask = voice_enhance_mask[:, np.newaxis]
        mag_clean = mag_clean * voice_enhance_mask
        
        # Enhanced voice frequency boosting
        formant_mask = (freqs >= 300) & (freqs <= 3000)
        consonant_mask = (freqs >= 3000) & (freqs <= 4000)
        presence_mask = (freqs >= 2000) & (freqs <= 5000)  # New presence boost
        
        # Stronger boosting for clarity and presence
        mag_clean[formant_mask] *= 1.4     # Even stronger formant boost
        mag_clean[consonant_mask] *= 1.3   # Stronger consonant boost
        mag_clean[presence_mask] *= 1.25   # Additional presence boost
        
        # Additional processing for better voice clarity
        # Adaptive smoothing based on frequency
        for i in range(len(freqs)):
            if formant_mask[i] or consonant_mask[i]:
                # Minimal smoothing in voice frequencies
                mag_clean[i] = gaussian_filter1d(mag_clean[i], sigma=0.5)
            else:
                # More smoothing outside voice frequencies
                mag_clean[i] = gaussian_filter1d(mag_clean[i], sigma=2.0)
        
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
        
        # Mix in original signal with emphasis on voice frequencies
        mix_ratio = 0.20  # Increased mix ratio
        audio_clean = (1 - mix_ratio) * audio_clean + mix_ratio * float_data
        
        # Apply stronger volume boost
        boost_factor = 1.8  # 80% volume increase
        audio_clean = np.clip(audio_clean * boost_factor, -1.0, 1.0)
        
        # Enhanced dynamic range compression for louder voice
        threshold = 0.25  # Lower threshold to catch more of the signal
        ratio = 0.4      # Stronger compression (2.5:1)
        makeup_gain = 1.4  # More makeup gain
        
        # Apply compression
        audio_compressed = np.copy(audio_clean)
        mask = np.abs(audio_clean) > threshold
        audio_compressed[mask] = np.sign(audio_clean[mask]) * (
            threshold + (np.abs(audio_clean[mask]) - threshold) * ratio
        )
        
        # Apply makeup gain
        audio_compressed *= makeup_gain
        
        # Second stage compression for extra presence
        threshold2 = 0.4
        ratio2 = 0.6
        makeup_gain2 = 1.2
        
        mask2 = np.abs(audio_compressed) > threshold2
        audio_compressed[mask2] = np.sign(audio_compressed[mask2]) * (
            threshold2 + (np.abs(audio_compressed[mask2]) - threshold2) * ratio2
        )
        audio_compressed *= makeup_gain2
        
        # Final clip to prevent distortion
        audio_compressed = np.clip(audio_compressed, -1.0, 1.0)
        
        # Convert back to original format
        return denormalize_audio(audio_compressed)
        
    except Exception as e:
        print(f"Robust noise filtering error: {str(e)}")
        return data
