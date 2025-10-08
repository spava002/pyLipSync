# pyLipSync

A Python implementation of [Hecomi's uLipSync](https://github.com/hecomi/uLipSync) for audio-based lip sync analysis. This library analyzes audio and determines phoneme targets for lip synchronization in real-time applications.

## Installation

### Install from GitHub

You can install directly from the GitHub repository:

```bash
pip install git+https://github.com/spava002/pyLipSync.git
```

### Install from Local Clone

Alternatively, clone the repository and install:

```bash
git clone https://github.com/spava002/pyLipSync.git
cd pyLipSync
pip install -e .
```

### Install Dependencies via requirements.txt

If you prefer to install dependencies separately:

```bash
pip install -r requirements.txt
```

## ⚠️ Important: Audio Template Setup

**Before the library will work, you MUST add audio template files to the phoneme folders.**

The library includes pre-configured folders for basic phonemes:
- `pyLipSync/audio/aa/` - for "A" sounds
- `pyLipSync/audio/ee/` - for "E" sounds  
- `pyLipSync/audio/ih/` - for "I" sounds
- `pyLipSync/audio/oh/` - for "O" sounds
- `pyLipSync/audio/ou/` - for "U" sounds
- `pyLipSync/audio/silence/` - for silence

**You need to add your own audio samples (`.mp3`, `.wav`, `.ogg`, `.flac`, etc.) to these folders.** The library will use these samples to a build phoneme template for matching onto new incoming audio.

### Adding More Phonemes

To add additional phonemes:
1. Create a new folder inside `pyLipSync/audio/`
2. **The folder name becomes the phoneme name** (e.g., `pyLipSync/audio/th/` for "th" sounds)
3. Add audio sample files to the new folder

The library will automatically detect and use all phoneme folders when building templates.

## Usage

Basic usage example (see `main.py` for reference):

```python
import librosa as lb
from pyLipSync import LipSync, CompareMethod

# Initialize LipSync with your preferred comparison method
lipsync = LipSync(
    compare_method=CompareMethod.COSINE_SIMILARITY  # Options: L1_NORM, L2_NORM, COSINE_SIMILARITY
)

# Load your audio file
audio, sr = lb.load("path/to/your/audio.mp3", sr=None)

# Process audio and get phoneme segments
segments = lipsync.process_audio_segments(audio, sr)

# Get the most prominent phoneme for each segment
for segment in segments:
    max_phoneme = segment.get_max_target_phoneme()
    print(f"Phoneme: {max_phoneme.name}, Confidence: {max_phoneme.target:.2f}")
```

## Quick Start

1. **Install the package**:
   ```bash
   pip install git+https://github.com/spava002/pyLipSync.git
   ```

2. **Add audio templates** to `pyLipSync/audio/` folders (REQUIRED!)

3. **Run the example**:
   ```bash
   python main.py
   ```

## Credits

This is a Python implementation of [uLipSync](https://github.com/hecomi/uLipSync) by Hecomi. 

## License

MIT License - see [LICENSE](LICENSE) file for details.