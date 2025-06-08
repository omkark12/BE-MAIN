import numpy as np
from typing import Optional, Tuple, Dict
from .ml_models import MLAudioProcessor

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
    
    # Convert to float32 for processing
    try:
        float_data = normalize_audio(data)
    except Exception as e:
        print(f"Warning: Data normalization failed: {str(e)}")
        return data, noise_info
    
    if use_ml:
        try:
            ml_proc = get_ml_processor()
            # Analyze noise type
            noise_info = ml_proc.classify_noise(float_data, rate)
            
            # Apply ML-based denoising
            filtered_float = ml_proc.denoise_audio(float_data, rate, noise_info['noise_type'])
            
            # Convert back to int16
            return denormalize_audio(filtered_float), noise_info
            
        except Exception as e:
            print(f"Warning: ML-based filtering failed ({str(e)}), falling back to FFT filtering")
            return fft_filter(data, rate, intensity), noise_info
    
    # Fall back to FFT-based filtering
    return fft_filter(data, rate, intensity), noise_info
