import numpy as np
from dataclasses import dataclass, field

@dataclass
class Phoneme:
    name: str
    target: float

    def to_dict(self):
        return {"name": self.name, "target": self.target}

@dataclass
class PhonemeSegment:
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
    mfcc: list[float]
    volume: float = None
    normalized_volume: float = None
    phonemes: list[Phoneme] = field(default_factory=list)