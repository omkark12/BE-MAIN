from audio_processing.audio_io import AudioProcessor

def test_audio_io():
    processor = AudioProcessor()
    assert processor.rate == 44100
    assert processor.chunk == 1024
    print("Audio I/O tests passed.")
