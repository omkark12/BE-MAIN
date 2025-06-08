import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import librosa
import torch.nn as nn
import os
import urllib.request
import json
import hashlib
import time
import requests
from urllib.parse import urlparse

def ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Convert audio data to float32 in range [-1, 1]"""
    if audio.dtype == np.int16:
        return (audio.astype(np.float32) / 32768.0).astype(np.float32)
    elif audio.dtype == np.float64:
        return audio.astype(np.float32)
    elif audio.dtype == np.float32:
        return audio
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

class MLAudioProcessor:
    def __init__(self):
        print("Initializing ML Audio Processor...")
        
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models with local caching and proper error handling
        try:
            self.vad_model = self._init_vad_model()
            if self.vad_model is None:
                print("Warning: Using fallback energy-based VAD")
                
            self.noise_classifier = self._init_noise_classifier()
            if self.noise_classifier is None:
                print("Warning: Using fallback noise classifier")
                
            self.denoiser = self._init_denoiser()
            if self.denoiser is None:
                print("Warning: Using fallback FFT-based denoising")
        except Exception as e:
            print(f"Error during model initialization: {str(e)}")
            # Ensure we have at least fallback models
            self.vad_model = ThresholdVAD()
            self.noise_classifier = SimpleNoiseClassifier()
            self.denoiser = SpectralDenoiser()

        print("ML Audio Processor initialization complete")

    def _download_with_retry(self, url: str, filepath: str, max_retries: int = 3) -> bool:
        """Download a file with retry logic"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()
                
                # Write the file in chunks
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        return False

    def _init_vad_model(self):
        """Initialize a simple threshold-based VAD"""
        try:
            return ThresholdVAD()
        except Exception as e:
            print(f"Error initializing VAD: {str(e)}")
            return None

    def _init_noise_classifier(self):
        """Initialize a simple noise classifier"""
        try:
            return SimpleNoiseClassifier()
        except Exception as e:
            print(f"Error initializing noise classifier: {str(e)}")
            return None

    def _init_denoiser(self):
        """Initialize spectral denoiser"""
        try:
            return SpectralDenoiser()
        except Exception as e:
            print(f"Error initializing denoiser: {str(e)}")
            return None

    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int,
                            threshold: float = 0.5) -> np.ndarray:
        """
        Perform Voice Activity Detection.
        
        Args:
            audio: Input audio signal (int16 or float32)
            sample_rate: Sampling rate of the audio
            threshold: VAD threshold (0-1)
            
        Returns:
            Binary mask of voice activity
        """
        # Convert to float32 for processing
        float_audio = ensure_float32(audio)
        
        if self.vad_model is None:
            return self._energy_based_vad(float_audio, sample_rate, threshold)

        try:
            return self.vad_model.detect(float_audio, sample_rate, threshold)
        except Exception as e:
            print(f"Warning: VAD failed ({str(e)}), falling back to energy-based VAD")
            return self._energy_based_vad(float_audio, sample_rate, threshold)

    def _energy_based_vad(self, audio: np.ndarray, sample_rate: int,
                         threshold: float = 1e-4) -> np.ndarray:
        """Energy-based VAD implementation for fallback"""
        # Ensure float32
        audio = ensure_float32(audio)
        
        frame_duration_ms = 30
        frame_length = int(sample_rate * frame_duration_ms / 1000)
        
        if len(audio) < frame_length:
            return np.ones_like(audio, dtype=bool)

        # Calculate energy for each frame
        num_frames = len(audio) // frame_length
        energies = np.array([
            np.sum(frame ** 2) / len(frame)
            for frame in np.array_split(audio[:num_frames * frame_length], num_frames)
        ])
        
        # Create mask
        mask = energies > threshold
        # Expand mask to match audio length
        expanded_mask = np.repeat(mask, frame_length)
        # Pad if necessary
        if len(expanded_mask) < len(audio):
            expanded_mask = np.pad(expanded_mask,
                                 (0, len(audio) - len(expanded_mask)),
                                 'edge')
        return expanded_mask

    def classify_noise(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Classify the type of noise in the audio segment.
        
        Args:
            audio: Input audio signal (int16 or float32)
            sample_rate: Sampling rate of the audio
            
        Returns:
            Dictionary containing noise type and confidence score
        """
        # Convert to float32 for processing
        float_audio = ensure_float32(audio)
        
        if self.noise_classifier is None:
            return {'noise_type': 'unknown', 'confidence': 0.0}

        try:
            return self.noise_classifier.classify(float_audio, sample_rate)
        except Exception as e:
            print(f"Warning: Noise classification failed: {str(e)}")
            return {'noise_type': 'unknown', 'confidence': 0.0}

    def denoise_audio(self, audio: np.ndarray, sample_rate: int,
                     noise_type: Optional[str] = None) -> np.ndarray:
        """
        Perform noise reduction using spectral subtraction.
        
        Args:
            audio: Input audio signal (int16 or float32)
            sample_rate: Sampling rate of the audio
            noise_type: Optional noise type for adaptive filtering
            
        Returns:
            Denoised audio signal (same type as input)
        """
        # Convert to float32 for processing
        float_audio = ensure_float32(audio)
        original_dtype = audio.dtype
        
        if self.denoiser is None:
            return audio

        try:
            denoised = self.denoiser.denoise(float_audio, sample_rate)
            # Return in the same format as input
            if original_dtype == np.int16:
                return (denoised * 32767).astype(np.int16)
            return denoised
        except Exception as e:
            print(f"Warning: Denoising failed: {str(e)}")
            return audio

class ThresholdVAD:
    """Advanced threshold-based VAD using multiple features"""
    def __init__(self):
        # Optimize frame size for faster processing
        self.frame_length = 512  # Increased for better frequency resolution
        self.hop_length = 256    # 50% overlap
        
        # Pre-compute constants
        self.window = np.hanning(self.frame_length).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.frame_length).astype(np.float32)
        
        # Cache for feature normalization
        self.feature_means = None
        self.feature_ranges = None
        
        # Hysteresis parameters
        self.activation_threshold = 0.4
        self.deactivation_threshold = 0.3
        self.smoothing_window = 3  # Reduced for lower latency

    def detect(self, audio: np.ndarray, sample_rate: int, threshold: float) -> np.ndarray:
        try:
            # Ensure float32 and proper shape
            audio = ensure_float32(audio)
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Quick energy check for silence
            if np.max(np.abs(audio)) < 1e-4:
                return np.zeros_like(audio, dtype=bool)

            # Extract frames efficiently using librosa
            frames = librosa.util.frame(
                audio,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            ).T.astype(np.float32)  # Shape: (n_frames, frame_length)
            
            if len(frames) == 0:
                print("Warning: No frames extracted")
                return np.ones_like(audio, dtype=bool)
            
            # Apply window
            frames = frames * self.window
            
            # Compute spectrogram
            spec = np.abs(np.fft.rfft(frames)).astype(np.float32)  # Shape: (n_frames, n_freqs)
            spec = np.maximum(spec, 1e-10)
            
            # Calculate features
            energies = np.sum(spec ** 2, axis=1)
            
            # Spectral centroid
            freqs = self.freq_bins[np.newaxis, :] * sample_rate
            spec_sum = np.sum(spec, axis=1, keepdims=True)
            spectral_centroid = np.sum(freqs * spec, axis=1) / (spec_sum.squeeze() + 1e-6)
            
            # Spectral flatness
            log_geometric_mean = np.mean(np.log(spec + 1e-10), axis=1)
            arithmetic_mean = np.mean(spec, axis=1)
            spectral_flatness = np.exp(log_geometric_mean) / (arithmetic_mean + 1e-6)
            
            # Zero-crossing rate
            zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
            
            # Normalize features
            features = np.stack([
                energies,
                spectral_centroid,
                spectral_flatness,
                zcr
            ])
            
            if self.feature_means is None:
                self.feature_means = np.mean(features, axis=1, keepdims=True)
                self.feature_ranges = np.ptp(features, axis=1, keepdims=True)
                self.feature_ranges[self.feature_ranges == 0] = 1
            
            # Normalize
            features_norm = (features - self.feature_means) / (self.feature_ranges + 1e-6)
            
            # Combine features
            voice_prob = (
                0.4 * features_norm[0] +  # energy
                0.3 * features_norm[1] +  # spectral centroid
                0.2 * (1 - features_norm[2]) +  # inverse spectral flatness
                0.1 * features_norm[3]  # zcr
            )
            
            # Apply hysteresis
            mask = np.zeros_like(voice_prob, dtype=bool)
            mask[0] = voice_prob[0] > self.activation_threshold
            mask[1:] = np.where(
                mask[:-1],
                voice_prob[1:] > self.deactivation_threshold,
                voice_prob[1:] > self.activation_threshold
            )
            
            # Smooth mask
            if self.smoothing_window > 1:
                kernel = np.ones(self.smoothing_window) / self.smoothing_window
                smooth_mask = np.convolve(mask.astype(float), kernel, mode='same')
                mask = smooth_mask > 0.5
            
            # Interpolate back to audio length
            hop_times = np.arange(len(mask)) * self.hop_length
            audio_times = np.arange(len(audio))
            
            mask_interp = np.interp(
                audio_times,
                hop_times,
                mask.astype(float)
            )
            
            return mask_interp > 0.5
            
        except Exception as e:
            print(f"Warning: VAD processing failed: {str(e)}")
            print(f"Debug info - audio shape: {audio.shape}, dtype: {audio.dtype}")
            if 'frames' in locals():
                print(f"Frames shape: {frames.shape}")
            if 'spec' in locals():
                print(f"Spectrogram shape: {spec.shape}")
            return np.ones_like(audio, dtype=bool)

class SimpleNoiseClassifier:
    """Simple noise classifier using spectral features"""
    def __init__(self):
        self.noise_classes = [
            'background_noise', 'speech', 'music', 'machine', 'traffic',
            'nature', 'animal', 'human', 'impact', 'other'
        ]
        # Adjust frame parameters
        self.n_fft = 1024  # Match chunk size
        self.hop_length = 256  # Reduced for better overlap
        self.n_mels = 64  # Reduced for faster processing

    def classify(self, audio: np.ndarray, sample_rate: int) -> dict:
        try:
            # Ensure input length is sufficient
            if len(audio) < self.n_fft:
                return {'noise_type': 'unknown', 'confidence': 0.0}
                
            # Extract features with adjusted parameters
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                center=False  # Disable centering
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Extract features with adjusted parameters
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio, 
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False
            ))
            
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                y=audio, 
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False
            ))
            
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(
                audio,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
                center=False
            ))
            
            # Enhanced classification logic
            if zero_crossing_rate > 0.15:
                if spectral_centroid > 3000:
                    noise_type = 'machine'
                    confidence = 0.8
                else:
                    noise_type = 'speech'
                    confidence = 0.7
            elif spectral_rolloff < 1000:
                if zero_crossing_rate < 0.05:
                    noise_type = 'background_noise'
                    confidence = 0.75
                else:
                    noise_type = 'nature'
                    confidence = 0.6
            elif 1000 <= spectral_centroid <= 2000:
                if zero_crossing_rate > 0.1:
                    noise_type = 'human'
                    confidence = 0.65
                else:
                    noise_type = 'music'
                    confidence = 0.7
            else:
                noise_type = 'other'
                confidence = 0.5
            
            return {
                'noise_type': noise_type,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Warning: Feature extraction failed: {str(e)}")
            return {
                'noise_type': 'unknown',
                'confidence': 0.0
            }

class SpectralDenoiser:
    """Balanced spectral denoiser with adaptive noise estimation"""
    def __init__(self):
        # FFT parameters
        self.n_fft = 2048  # Back to larger FFT for better frequency resolution
        self.hop_length = 512
        self.win_length = 2048
        
        # Pre-compute constants
        self.window = np.hanning(self.win_length).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(self.n_fft, 1/44100).astype(np.float32)
        
        # Voice and noise parameters
        self.voice_freq_range = (50, 8000)  # Wider range for better voice capture
        self.noise_estimation_percentile = 40  # More conservative noise estimation
        
        # Adaptive thresholds
        self.noise_floor_min = 1e-4
        self.noise_floor_max = 1.0
        self.noise_reduction_strength = 0.6  # Moderate reduction
        
        # Smoothing parameters
        self.temporal_smoothing_factor = 0.7
        self.freq_smoothing_width = 4
        
        # Initialize noise profile
        self.noise_profile = None
        self.noise_profile_weight = 0.7
        
    def _estimate_noise_profile(self, mag):
        """Estimate noise profile using statistical methods"""
        if mag.shape[1] < 4:  # Need minimum frames for estimation
            return np.mean(mag, axis=1, keepdims=True)
            
        # Sort magnitudes along time axis
        sorted_mag = np.sort(mag, axis=1)
        # Use lowest 20% of magnitudes for initial estimate
        n_lowest = max(1, int(0.2 * mag.shape[1]))
        noise_estimate = np.mean(sorted_mag[:, :n_lowest], axis=1, keepdims=True)
        
        return noise_estimate
        
    def _smooth_mask(self, mask):
        """Apply 2D smoothing to the mask"""
        # Temporal smoothing
        smoothed = np.zeros_like(mask)
        alpha = self.temporal_smoothing_factor
        
        smoothed[:, 0] = mask[:, 0]
        for i in range(1, mask.shape[1]):
            smoothed[:, i] = alpha * smoothed[:, i-1] + (1-alpha) * mask[:, i]
            
        # Frequency smoothing
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(smoothed, sigma=1.5, axis=0)
        
        return smoothed

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            # Input validation and preprocessing
            audio = ensure_float32(audio)
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            if len(audio) < self.n_fft:
                return audio
                
            # Compute STFT
            D = librosa.stft(audio,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length,
                           window=self.window,
                           center=True)  # Enable centering for better edge handling
            
            # Compute magnitude and phase
            mag = np.abs(D)
            phase = np.angle(D)
            
            # Estimate current noise profile
            current_noise_profile = self._estimate_noise_profile(mag)
            
            # Update running noise profile
            if self.noise_profile is None:
                self.noise_profile = current_noise_profile
            else:
                self.noise_profile = (self.noise_profile_weight * self.noise_profile + 
                                    (1 - self.noise_profile_weight) * current_noise_profile)
            
            # Compute adaptive threshold based on frequency
            freq_weights = np.ones_like(self.freq_bins)
            voice_mask = ((self.freq_bins >= self.voice_freq_range[0]) & 
                         (self.freq_bins <= self.voice_freq_range[1]))
            freq_weights[voice_mask] *= 0.6  # Less reduction in voice range
            freq_weights = freq_weights[:, np.newaxis]
            
            # Calculate noise reduction mask
            noise_mask = (mag - self.noise_reduction_strength * freq_weights * self.noise_profile)
            noise_mask = noise_mask / (mag + self.noise_floor_min)
            noise_mask = np.maximum(0, noise_mask)
            noise_mask = np.minimum(1, noise_mask)
            
            # Apply smoothing
            noise_mask = self._smooth_mask(noise_mask)
            
            # Apply mask and reconstruct
            mag_denoised = mag * noise_mask
            D_denoised = mag_denoised * np.exp(1j * phase)
            
            # Inverse STFT
            audio_denoised = librosa.istft(D_denoised,
                                         hop_length=self.hop_length,
                                         win_length=self.win_length,
                                         window=self.window,
                                         center=True)
            
            # Ensure output length matches input
            if len(audio_denoised) > len(audio):
                audio_denoised = audio_denoised[:len(audio)]
            elif len(audio_denoised) < len(audio):
                audio_denoised = np.pad(audio_denoised, (0, len(audio) - len(audio_denoised)))
            
            # Mix in a small amount of the original signal
            mix_ratio = 0.15
            audio_denoised = (1 - mix_ratio) * audio_denoised + mix_ratio * audio
            
            return audio_denoised.astype(np.float32)
            
        except Exception as e:
            print(f"Denoising error: {str(e)}")
            return audio 