import pytest
import numpy as np
from pylipsync import LipSync, CompareMethod

class TestValidation:
    """Test that bad inputs are caught."""
    
    def test_invalid_silence_threshold(self):
        """Test silence threshold validation."""
        with pytest.raises(ValueError):
            LipSync(silence_threshold=1.5)
        with pytest.raises(ValueError):
            LipSync(silence_threshold=-0.1)
    
    def test_empty_audio_rejected(self, lipsync: LipSync):
        """Test that empty audio is rejected."""
        with pytest.raises(ValueError):
            lipsync.process_audio_segments(np.array([]), 16000)
    
    def test_invalid_sample_rate(self, lipsync: LipSync, simple_sine_wave: tuple[np.ndarray, int]):
        """Test that invalid sample rates are rejected."""
        audio, _ = simple_sine_wave
        with pytest.raises(ValueError):
            lipsync.process_audio_segments(audio, -1)
        with pytest.raises(ValueError):
            lipsync.process_audio_segments(audio, 0)
        with pytest.raises(ValueError):
            lipsync.process_audio_segments(audio, 8000)
    
    def test_invalid_audio_dimensions(self, lipsync: LipSync):
        """Test that 2D audio is rejected."""
        audio_2d = np.zeros((2, 1000))
        with pytest.raises(ValueError):
            lipsync.process_audio_segments(audio_2d, 16000)


class TestCoreProcessing:
    """Test the main audio processing functionality."""
    
    def test_process_audio_segments(self, lipsync: LipSync, simple_sine_wave: tuple[np.ndarray, int]):
        """Test that audio processing returns valid segments."""
        audio, sr = simple_sine_wave
        segments = lipsync.process_audio_segments(audio, sr)
        
        # Basic sanity checks
        assert len(segments) > 0
        assert all(len(seg.phonemes) > 0 for seg in segments)
        assert all(seg.volume >= 0 for seg in segments)
        assert all(0 <= seg.normalized_volume <= 1 for seg in segments)
    
    def test_silence_detection(self, lipsync: LipSync, silence_audio: tuple[np.ndarray, int]):
        """Test that silence is properly detected."""
        audio, sr = silence_audio
        segments = lipsync.process_audio_segments(audio, sr)
        
        # Most segments should be silence
        silence_count = sum(1 for seg in segments if seg.is_silence())
        assert silence_count > len(segments) * 0.8  # At least 80% silence
    
    def test_comparison_methods_all_work(self, simple_sine_wave: tuple[np.ndarray, int]):
        """Test that all comparison methods produce results."""
        audio, sr = simple_sine_wave
        
        compare_methods = [CompareMethod.L1_NORM, CompareMethod.L2_NORM, CompareMethod.COSINE_SIMILARITY]
        for method in compare_methods:
            lipsync = LipSync(compare_method=method)
            segments = lipsync.process_audio_segments(audio, sr)
            assert len(segments) > 0


class TestIntegration:
    """End to end test with real package audio."""
    
    def test_with_package_audio_file(self, lipsync: LipSync):
        """Test processing actual phoneme audio from the package."""
        import os, librosa
        
        lipsync = LipSync()

        package_dir = os.path.dirname(os.path.abspath(__file__))
        audio_path = os.path.join(package_dir, "audio", "aa", "A_female.mp3")
        
        if os.path.exists(audio_path):
            audio, sr = librosa.load(audio_path, sr=None)
            segments = lipsync.process_audio_segments(audio, sr)
            
            assert len(segments) > 0
            
            # Should predominantly detect "aa" phoneme
            non_silence = [s for s in segments if not s.is_silence()]
            if non_silence:
                phoneme_counts = {}
                for seg in non_silence:
                    name = seg.most_prominent_phoneme().name
                    phoneme_counts[name] = phoneme_counts.get(name, 0) + 1
                
                # "aa" should be most common
                most_common = max(phoneme_counts, key=phoneme_counts.get)
                assert most_common == "aa", f"Expected 'aa', got '{most_common}'"