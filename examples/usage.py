"""
Basic example showing how to use pyLipSync to analyze audio.
"""

import os
import librosa as lb
from pyLipSync import LipSync, CompareMethod

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

def main(example_audio_path: str):
    lipsync = LipSync(
        compare_method=CompareMethod.COSINE_SIMILARITY # Default is CosineSimilarity but you can also choose from L1Norm and L2Norm
    )

    # Load in a test audio file to generate phoneme segments for
    audio, sr = lb.load(os.path.join(PACKAGE_DIR, example_audio_path), sr=None)

    segments = lipsync.process_audio_segments(audio, sr)

    # View each segment's data
    for segment in segments:
        most_prominent_phoneme = segment.most_prominent_phoneme() if not segment.is_silence() else None
        print(f"Most Prominent Phoneme: {most_prominent_phoneme}")

if __name__ == "__main__":
    main("audio/example_female1.mp3")