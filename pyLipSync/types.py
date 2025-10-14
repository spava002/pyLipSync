import numpy as np
from dataclasses import dataclass, field

@dataclass
class Phoneme:
    """
    Represents a single phoneme with a target value.
    
    Attributes:
        name: The name of the phoneme
        target: The target value for the phoneme
    """
    name: str
    target: float

    def to_dict(self):
        return {"name": self.name, "target": self.target}

@dataclass
class PhonemeSegment:
    """
    Represents a single audio segment with phoneme analysis results.
    
    Attributes:
        start_time: Start time of segment in seconds
        end_time: End time of segment in seconds
        volume: RMS volume of the audio segment
        normalized_volume: Volume normalized to 0-1 range
        audio: Raw audio data for this segment
        phonemes: List of detected phonemes with confidence scores
    """
    start_time: float
    end_time: float
    volume: float
    normalized_volume: float
    audio: np.ndarray
    phonemes: list[Phoneme]

    def to_dict(self):
        return {"audio": self.audio.tobytes(), "phonemes": [phoneme.to_dict() for phoneme in self.phonemes]}

    def most_prominent_phoneme(self) -> Phoneme:
        return max(self.phonemes, key=lambda phoneme: phoneme.target)

    def is_silence(self) -> bool:
        return all(phoneme.target == 0 for phoneme in self.phonemes)

@dataclass
class LipSyncInfo:
    """
    Represents the results of the lip sync analysis.
    
    Attributes:
        mfcc: The MFCC of the audio
        volume: The RMS volume of the audio
        normalized_volume: The normalized volume of the audio
        phonemes: The detected phonemes with confidence scores
    """
    mfcc: list[float]
    volume: float = None
    normalized_volume: float = None
    phonemes: list[Phoneme] = field(default_factory=list)