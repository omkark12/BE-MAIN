import pyaudio
import numpy as np
from audio_processing.noise_filter import filter_noise
from audio_processing.stft_vad import vad_energy
from audio_processing.ml_models import MLAudioProcessor
import threading
import queue
import time

class AudioProcessor:
    def __init__(self):
        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.running = False
        
        try:
            self.audio = pyaudio.PyAudio()
            # Print available devices
            print("\nAvailable audio devices:")
            for i in range(self.audio.get_device_count()):
                dev_info = self.audio.get_device_info_by_index(i)
                print(f"Device {i}: {dev_info['name']}")
                print(f"  Max Input Channels: {dev_info['maxInputChannels']}")
                print(f"  Max Output Channels: {dev_info['maxOutputChannels']}")
                print(f"  Default Sample Rate: {dev_info['defaultSampleRate']}")
            print()
            
            # Find default devices
            self.input_device = self.audio.get_default_input_device_info()
            self.output_device = self.audio.get_default_output_device_info()
            print(f"Using input device: {self.input_device['name']}")
            print(f"Using output device: {self.output_device['name']}")
            
        except Exception as e:
            print(f"Error initializing audio: {str(e)}")
            raise
        
        # Initialize ML processor
        self.ml_processor = MLAudioProcessor()
        
        # Processing state
        self.is_voice_active_flag = False
        self.current_noise_info = {'noise_type': 'unknown', 'confidence': 0.0}
        
        # Processing queues
        self.audio_queue = queue.Queue(maxsize=8)
        self.result_queue = queue.Queue(maxsize=8)
        
        # Processing threads
        self.processing_thread = None
        
        # Processing state
        self.last_vad_result = False
        self.vad_smoothing = 0.7
        self.noise_update_interval = 0.5
        self.last_noise_update = 0
        
        # Buffer for overlap processing
        self.overlap_buffer = np.zeros(self.chunk, dtype=np.float32)

    def start_stream(self, intensity):
        """Start audio processing with the given noise reduction intensity"""
        if self.running:
            print("Warning: Audio processing already running")
            return
            
        try:
            print("Starting audio processing...")
            # Reset state
            self.running = True
            self.audio_queue = queue.Queue(maxsize=8)
            self.result_queue = queue.Queue(maxsize=8)
            self.overlap_buffer = np.zeros(self.chunk, dtype=np.float32)
            
            # Start processing thread
            print("Starting processing thread...")
            self.processing_thread = threading.Thread(target=self._processing_loop, 
                                                    args=(intensity,),
                                                    daemon=True)
            self.processing_thread.start()
            
            # Wait briefly to ensure thread starts
            time.sleep(0.1)
            
            if not self.processing_thread.is_alive():
                raise RuntimeError("Processing thread failed to start")
            
            print("Processing thread started successfully")
            print("Starting audio I/O...")
            
            # Start audio I/O in the main thread
            self._audio_io_loop(intensity)
            
        except Exception as e:
            print(f"Error starting audio processing: {str(e)}")
            self.running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            self.processing_thread = None

    def stop_stream(self):
        """Stop audio processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.audio.terminate()

    def _audio_io_loop(self, intensity):
        """Handle audio I/O with minimal latency"""
        input_stream = None
        output_stream = None
        chunks_processed = 0
        last_log_time = time.time()
        
        try:
            # Initialize audio streams with error checking
            print("Opening input stream...")
            try:
                input_stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    input_device_index=self.input_device['index'],
                    frames_per_buffer=self.chunk,
                    stream_callback=None
                )
                print("Input stream opened successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to open input stream: {str(e)}")
                
            print("Opening output stream...")
            try:
                output_stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    output=True,
                    output_device_index=self.output_device['index'],
                    frames_per_buffer=self.chunk,
                    stream_callback=None
                )
                print("Output stream opened successfully")
            except Exception as e:
                if input_stream:
                    input_stream.stop_stream()
                    input_stream.close()
                raise RuntimeError(f"Failed to open output stream: {str(e)}")

            error_count = 0
            max_errors = 5
            
            print("Starting audio processing loop...")
            while self.running:
                try:
                    # Print status every second
                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        print(f"Processing active - chunks processed: {chunks_processed}, queue sizes: in={self.audio_queue.qsize()}, out={self.result_queue.qsize()}")
                        last_log_time = current_time
                    
                    # Check if processing thread is still alive
                    if not self.processing_thread.is_alive():
                        raise RuntimeError("Processing thread died unexpectedly")
                    
                    # Check stream states
                    if not input_stream.is_active() or not output_stream.is_active():
                        raise RuntimeError("Audio stream(s) became inactive")
                    
                    # Read audio data
                    try:
                        data = input_stream.read(self.chunk, exception_on_overflow=False)
                    except Exception as e:
                        print(f"Error reading audio data: {str(e)}")
                        continue
                        
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Add to processing queue
                    try:
                        self.audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        print("Warning: Processing queue full, skipping chunk")
                        pass
                    
                    # Get processed audio
                    try:
                        processed_audio = self.result_queue.get(timeout=0.01)
                        output_stream.write(processed_audio.tobytes())
                        chunks_processed += 1
                        error_count = 0  # Reset error count on successful processing
                    except queue.Empty:
                        # Use original audio if no processed data available
                        output_stream.write(data)
                        print("Warning: No processed audio available, using original")
                        
                except Exception as e:
                    error_count += 1
                    print(f"Warning: Audio I/O error ({error_count}/{max_errors}): {str(e)}")
                    if error_count >= max_errors:
                        raise RuntimeError(f"Too many consecutive errors: {str(e)}")
                    time.sleep(0.1)  # Brief pause before retrying

        except Exception as e:
            print(f"Fatal error in audio I/O loop: {str(e)}")
            self.running = False
            
        finally:
            print(f"Audio I/O loop ending - processed {chunks_processed} chunks")
            print("Cleaning up audio streams...")
            # Clean up audio streams
            if input_stream:
                try:
                    input_stream.stop_stream()
                    input_stream.close()
                    print("Input stream closed")
                except Exception as e:
                    print(f"Warning: Error closing input stream: {str(e)}")
                    
            if output_stream:
                try:
                    output_stream.stop_stream()
                    output_stream.close()
                    print("Output stream closed")
                except Exception as e:
                    print(f"Warning: Error closing output stream: {str(e)}")
                    
            # Stop processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                print("Stopping processing thread...")
                self.running = False
                self.processing_thread.join(timeout=1.0)
                print("Processing thread stopped")

    def _process_chunk(self, chunk, intensity):
        """Process a single audio chunk"""
        try:
            process_start = time.time()
            
            # Convert to float32
            float_data = self._normalize_audio(chunk)
            normalize_time = time.time() - process_start
            
            # Combine with overlap buffer
            audio_data = np.concatenate([self.overlap_buffer, float_data])
            concat_time = time.time() - process_start - normalize_time
            
            # Update VAD
            vad_start = time.time()
            try:
                vad_mask = self.ml_processor.detect_voice_activity(audio_data, self.rate)
                current_vad = np.mean(vad_mask) > 0.3
                self.is_voice_active_flag = (self.vad_smoothing * self.last_vad_result + 
                                           (1 - self.vad_smoothing) * current_vad)
                self.last_vad_result = self.is_voice_active_flag
                if current_vad != self.last_vad_result:
                    print(f"VAD state changed: {'active' if self.is_voice_active_flag else 'inactive'}")
            except Exception as e:
                print(f"Warning: VAD failed: {str(e)}")
                vad_mask = None
            vad_time = time.time() - vad_start
            
            # Update noise classification periodically
            current_time = time.time()
            noise_time = 0
            if current_time - self.last_noise_update >= self.noise_update_interval:
                noise_start = time.time()
                try:
                    new_noise_info = self.ml_processor.classify_noise(audio_data, self.rate)
                    if new_noise_info['noise_type'] != self.current_noise_info.get('noise_type'):
                        print(f"Noise type changed: {new_noise_info['noise_type']} (confidence: {new_noise_info['confidence']:.2f})")
                    self.current_noise_info = new_noise_info
                except Exception as e:
                    print(f"Warning: Noise classification failed: {str(e)}")
                self.last_noise_update = current_time
                noise_time = time.time() - noise_start
            
            # Apply noise reduction
            denoise_start = time.time()
            try:
                # Use ML-based denoising directly
                filtered_data = self.ml_processor.denoise_audio(
                    audio_data, 
                    self.rate,
                    self.current_noise_info.get('noise_type')
                )
                
                # Apply VAD mask if available
                if vad_mask is not None:
                    filtered_data = filtered_data * vad_mask
                
                # Update overlap buffer and get current chunk
                self.overlap_buffer = filtered_data[-self.chunk:]
                output_data = filtered_data[:self.chunk]
                
                # Convert back to int16
                result = self._denormalize_audio(output_data)
                
                # Log processing times periodically
                if current_time - self.last_noise_update >= self.noise_update_interval:
                    total_time = time.time() - process_start
                    print(f"Processing times - normalize: {normalize_time*1000:.1f}ms, "
                          f"concat: {concat_time*1000:.1f}ms, VAD: {vad_time*1000:.1f}ms, "
                          f"noise: {noise_time*1000:.1f}ms, denoise: {(time.time()-denoise_start)*1000:.1f}ms, "
                          f"total: {total_time*1000:.1f}ms")
                
                return result
                
            except Exception as e:
                print(f"Warning: Processing failed: {str(e)}")
                return chunk
                
        except Exception as e:
            print(f"Warning: Chunk processing failed: {str(e)}")
            return chunk

    def _processing_loop(self, intensity):
        """Audio processing loop"""
        print("Processing loop started")
        chunks_processed = 0
        last_log_time = time.time()
        last_chunk_time = time.time()
        max_processing_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Print status every second
                if current_time - last_log_time >= 1.0:
                    print(f"Processing thread stats - chunks: {chunks_processed}, max processing time: {max_processing_time*1000:.1f}ms")
                    print(f"Queue sizes - input: {self.audio_queue.qsize()}, output: {self.result_queue.qsize()}")
                    last_log_time = current_time
                
                # Get audio data with timeout
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    last_chunk_time = time.time()
                except queue.Empty:
                    # Check if we've been waiting too long for data
                    if time.time() - last_chunk_time > 1.0:
                        print("Warning: No audio data received for >1 second")
                    continue
                
                # Process chunk and measure time
                chunk_start = time.time()
                processed_data = self._process_chunk(audio_data, intensity)
                processing_time = time.time() - chunk_start
                max_processing_time = max(max_processing_time, processing_time)
                
                # Add to output queue
                try:
                    self.result_queue.put_nowait(processed_data)
                    chunks_processed += 1
                except queue.Full:
                    print("Warning: Output queue full, dropping processed chunk")
                    
            except Exception as e:
                print(f"Warning: Processing loop error: {str(e)}")
                # Brief pause to avoid tight loop on error
                time.sleep(0.1)
                
        print(f"Processing loop stopped - processed {chunks_processed} chunks")

    def _normalize_audio(self, audio_data):
        """Convert audio data to float32 in range [-1, 1]"""
        if audio_data.dtype == np.int16:
            return audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.float32:
            return np.clip(audio_data, -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported audio data type: {audio_data.dtype}")

    def _denormalize_audio(self, audio_data):
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

    def is_active(self):
        """Check if audio processing is active"""
        return self.running

    def is_voice_active(self):
        """Check if voice activity is detected"""
        return bool(self.is_voice_active_flag)

    def get_noise_info(self):
        """Get current noise classification"""
        return self.current_noise_info.copy()
