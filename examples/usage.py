"""
Basic example showing how to use pyLipSync to analyze audio.
"""

import librosa as lb
from pylipsync import LipSync, CompareMethod

lipsync = LipSync(
    compare_method=CompareMethod.COSINE_SIMILARITY # Default is CosineSimilarity but you can also choose from L1Norm and L2Norm
)

# Load in a test audio file to generate phoneme segments for
audio, sr = lb.load("path/to/your/audio.mp3", sr=None)

segments = lipsync.process_audio_segments(audio, sr)

# View each segment's data
for segment in segments:
    most_prominent_phoneme = segment.most_prominent_phoneme() if not segment.is_silence() else None
    print(f"({segment.start_time:.4f}-{segment.end_time:.4f})s | Most Prominent Phoneme: {most_prominent_phoneme}")