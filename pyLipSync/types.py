import numpy as np
from enum import Enum
from dataclasses import dataclass, field

# name = the actual phoneme name
# value = the string name of the folder containing the audio files for the phoneme
class PhonemeName(Enum):
    A = "aa"
    E = "ee"
    I = "ih"
    O = "oh"
    U = "ou"
    SILENCE = "silence"

@dataclass
class Phoneme:
    name: PhonemeName
    target: float

    def to_dict(self):
        return {"name": self.name.value, "target": self.target}

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