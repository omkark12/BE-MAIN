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
            denoised = self.denoiser.denoise(float_audio, sample_rate, noise_type)
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
    """Enhanced noise classifier using multiple spectral features"""
    def __init__(self):
        self.noise_classes = [
            'background_noise',  # Ambient room noise
            'speech',           # Human speech
            'music',            # Musical sounds
            'machine',          # Mechanical/electronic sounds
            'traffic',          # Vehicle/road noise
            'impact',           # Sudden loud noises
            'white_noise'       # Constant spectrum noise
        ]
        
        # Optimized analysis parameters
        self.n_fft = 2048      # Larger FFT for better frequency resolution
        self.hop_length = 512  # Shorter hop for better temporal resolution
        self.n_mels = 80       # More mel bands for better frequency discrimination
        
        # Feature extraction parameters
        self.frame_length = 2048
        self.min_freq = 20
        self.max_freq = 8000
        
    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Extract comprehensive set of audio features"""
        features = {}
        
        try:
            # Basic spectral features
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.min_freq,
                fmax=self.max_freq
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Spectral features
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(
                y=audio, 
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            ))
            
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(
                y=audio,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            ))
            
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(
                y=audio,
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            ))
            
            features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(
                y=audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            ))
            
            # Temporal features
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(
                audio,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            ))
            
            # Energy features
            features['rms_energy'] = np.mean(librosa.feature.rms(
                y=audio,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            ))
            
            # Rhythm features
            tempo, _ = librosa.beat.beat_track(
                y=audio,
                sr=sample_rate,
                hop_length=self.hop_length
            )
            features['tempo'] = tempo
            
            # Mel-frequency features
            mfcc = librosa.feature.mfcc(
                S=librosa.power_to_db(mel_spec),
                n_mfcc=13
            )
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_var'] = np.var(mfcc, axis=1)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None

    def classify(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Classify noise type using extracted features
        """
        try:
            # Extract features
            features = self._extract_features(audio, sample_rate)
            if features is None:
                return {'noise_type': 'unknown', 'confidence': 0.0}
            
            # Simple rule-based classification
            confidence = 0.0
            noise_type = 'unknown'
            
            # White noise detection
            if (features['spectral_flatness'] > 0.6 and
                features['zero_crossing_rate'] > 0.4):
                noise_type = 'white_noise'
                confidence = min(1.0, features['spectral_flatness'])
            
            # Music detection
            elif (features['tempo'] > 50 and
                  features['spectral_rolloff'] > 3000 and
                  np.std(features['mfcc_mean']) > 2.0):
                noise_type = 'music'
                confidence = 0.8
            
            # Machine noise detection
            elif (features['spectral_flatness'] > 0.3 and
                  features['spectral_bandwidth'] < 2000 and
                  features['zero_crossing_rate'] < 0.3):
                noise_type = 'machine'
                confidence = 0.7
            
            # Traffic noise detection
            elif (features['spectral_centroid'] < 2000 and
                  features['spectral_rolloff'] < 4000 and
                  features['rms_energy'] > 0.1):
                noise_type = 'traffic'
                confidence = 0.6
            
            # Impact noise detection
            elif (features['zero_crossing_rate'] > 0.6 and
                  features['rms_energy'] > 0.3):
                noise_type = 'impact'
                confidence = 0.7
            
            # Speech detection (when it's not the main signal)
            elif (2000 < features['spectral_centroid'] < 3000 and
                  np.mean(features['mfcc_var']) > 1.5):
                noise_type = 'speech'
                confidence = 0.65
            
            # Background noise (default)
            else:
                noise_type = 'background_noise'
                confidence = 0.5
            
            return {
                'noise_type': noise_type,
                'confidence': float(confidence)
            }
            
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return {'noise_type': 'unknown', 'confidence': 0.0}

class SpectralDenoiser:
    """Robust spectral denoiser with dual-stage noise reduction"""
    def __init__(self):
        # Core parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048
        
        # Voice frequency ranges
        self.voice_freq_ranges = [
            (85, 255),    # Male fundamental
            (165, 400),   # Female fundamental
            (400, 2000),  # Main formants
            (2000, 3500)  # Consonants
        ]
        
        # Noise reduction parameters
        self.noise_params = {
            'background_noise': {'reduction': 0.95, 'threshold': 0.02},
            'music': {'reduction': 0.92, 'threshold': 0.02},
            'machine': {'reduction': 0.97, 'threshold': 0.01},
            'white_noise': {'reduction': 0.97, 'threshold': 0.01},
            'speech': {'reduction': 0.80, 'threshold': 0.05},
            'other': {'reduction': 0.90, 'threshold': 0.03}
        }
        
        # Enhancement factors
        self.voice_boost = {
            'formant': 1.2,
            'consonant': 1.15
        }
        
        # Smoothing parameters
        self.smooth_factor = 0.85
        
    def _get_noise_threshold(self, mag_spec):
        """Estimate noise threshold using percentile method"""
        return np.percentile(mag_spec, 20, axis=1, keepdims=True)
    
    def _apply_frequency_weighting(self, freqs):
        """Apply frequency-dependent weighting"""
        weights = np.ones_like(freqs)
        
        # Weight voice frequencies
        for low, high in self.voice_freq_ranges:
            mask = (freqs >= low) & (freqs <= high)
            if low == 400 and high == 2000:  # Formants
                weights[mask] = 0.3  # Preserve more
            elif low >= 2000:  # Consonants
                weights[mask] = 0.4
            else:  # Fundamentals
                weights[mask] = 0.35
        
        # More reduction outside voice range
        weights[freqs < 85] = 1.5
        weights[freqs > 3500] = 1.4
        
        return weights[:, np.newaxis]
    
    def denoise(self, audio: np.ndarray, sample_rate: int, noise_type: str = None) -> np.ndarray:
        try:
            # Input validation
            audio = ensure_float32(audio)
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            if len(audio) < self.n_fft:
                return audio
            
            # Get noise parameters
            params = self.noise_params.get(noise_type, self.noise_params['other'])
            
            # Compute STFT
            D = librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window='hann',
                center=True
            )
            
            # Get magnitude and phase
            mag_spec = np.abs(D)
            phase_spec = np.angle(D)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=self.n_fft)
            
            # Get frequency weights
            weights = self._apply_frequency_weighting(freqs)
            
            # Estimate noise threshold
            noise_thresh = self._get_noise_threshold(mag_spec)
            
            # Apply noise reduction
            gain = np.maximum(
                1.0 - params['reduction'] * weights * (noise_thresh / (mag_spec + 1e-10)),
                params['threshold']
            )
            
            # Smooth the gain
            smoothed_gain = np.zeros_like(gain)
            smoothed_gain[:, 0] = gain[:, 0]
            for i in range(1, gain.shape[1]):
                smoothed_gain[:, i] = (self.smooth_factor * smoothed_gain[:, i-1] + 
                                     (1 - self.smooth_factor) * gain[:, i])
            
            # Apply gain and enhance voice frequencies
            mag_spec_clean = mag_spec * smoothed_gain
            
            # Enhance formants and consonants
            formant_mask = (freqs >= 400) & (freqs <= 2000)
            consonant_mask = (freqs >= 2000) & (freqs <= 3500)
            
            mag_spec_clean[formant_mask] *= self.voice_boost['formant']
            mag_spec_clean[consonant_mask] *= self.voice_boost['consonant']
            
            # Reconstruct signal
            D_clean = mag_spec_clean * np.exp(1j * phase_spec)
            audio_denoised = librosa.istft(
                D_clean,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window='hann',
                center=True
            )
            
            # Match length
            if len(audio_denoised) > len(audio):
                audio_denoised = audio_denoised[:len(audio)]
            elif len(audio_denoised) < len(audio):
                audio_denoised = np.pad(audio_denoised, (0, len(audio) - len(audio_denoised)))
            
            return audio_denoised.astype(np.float32)
            
        except Exception as e:
            print(f"Denoising error: {str(e)}")
            return audio