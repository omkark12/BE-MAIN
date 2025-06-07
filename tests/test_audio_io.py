from audio_processing.stft_vad import vad_energy

def test_vad_output_shape():
    import numpy as np
    dummy_audio = np.random.randn(44100).astype(np.float32)
    vad_result = vad_energy(dummy_audio, 44100)
    assert vad_result.shape[0] == dummy_audio.shape[0], "Output shape mismatch"
    print("VAD test passed.")
