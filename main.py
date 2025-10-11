import librosa as lb
from pyLipSync import LipSync, CompareMethod

lipsync = LipSync(
    compare_method=CompareMethod.COSINE_SIMILARITY # Can choose from L1Norm, L2Norm, or CosineSimilarity for comparison
)

# Load in a test audio file to generate phoneme segments for
audio, sr = lb.load("pyLipSync/audio/oh/O_female.mp3", sr=None)

segments = lipsync.process_audio_segments(audio, sr)

# View the most prominent phoneme for each segment
for segment in segments:
    print((segment.get_max_target_phoneme(), segment.volume, segment.normalized_volume))