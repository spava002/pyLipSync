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
    audio: np.ndarray
    phonemes: list[Phoneme]

    def to_dict(self):
        return {"audio": self.audio.tobytes(), "phonemes": [phoneme.to_dict() for phoneme in self.phonemes]}

    def get_max_target_phoneme(self):
        return max(self.phonemes, key=lambda phoneme: phoneme.target)

@dataclass
class LipSyncInfo:
    mfcc: list[float]
    phonemes: list[Phoneme] = field(default_factory=list)