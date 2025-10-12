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

## Quick Start

The library comes with built-in audio templates for common phonemes, so you can start using it immediately:

```python
import librosa as lb
from pyLipSync import LipSync, CompareMethod

# Initialize LipSync - works out of the box with default templates
lipsync = LipSync(
    compare_method=CompareMethod.COSINE_SIMILARITY  # Options: L1_NORM, L2_NORM, COSINE_SIMILARITY
)

# Load your audio file
audio, sr = lb.load("path/to/your/audio.mp3", sr=None)

# Process audio and get phoneme segments
segments = lipsync.process_audio_segments(
    audio,
    sr,
    window_size_ms=64.0,  # Window size in milliseconds
    fps=60                # Frames per second for output
)

# Get the most prominent phoneme for each segment
for segment in segments:
    max_phoneme = segment.get_max_target_phoneme()
    print(f"Phoneme: {max_phoneme.name}, Confidence: {max_phoneme.target:.2f}")
```

## Default Phonemes

The library includes pre-configured phoneme templates for:
- `aa` - "A" sounds
- `ee` - "E" sounds
- `ih` - "I" sounds
- `oh` - "O" sounds
- `ou` - "U" sounds
- `silence` - silence/no speech

These templates are ready to use without any additional setup.

### Adding New Phonemes

To add additional phonemes (e.g., consonants like "th", "sh", "f"):

1. Create a folder with all your phoneme names (or expand off the existing audio/ folder)
   ```
   audio/
   ├── aa/
   ├── ee/
   ├── th/          # New phoneme!
   │   └── th_sound.mp3
   └── sh/          # Another new one!
       └── sh_sound.mp3
   ```

2. Add audio samples to each folder (`.mp3`, `.wav`, `.ogg`, `.flac`, etc.)

3. Use your custom templates:
   ```python
   lipsync = LipSync(
       audio_templates_path="/path/to/my_custom_audio" # Not necessary if expanding within the audio/ folder
   )
   ```

**Note:** The folder name becomes the phoneme identifier in the output.

### Audio Sample Guidelines

- **Format**: Any common audio format (MP3, WAV, OGG, FLAC, M4A, AAC, AIFF)
- **Duration**: Short samples (0.5-2 seconds) work best
- **Quality**: Clear pronunciation without background noise
- **Multiple samples**: You can add multiple files per phoneme for better accuracy

## Advanced Configuration

```python
lipsync = LipSync(
    phoneme_templates_path="phonemes.json",      # Path to save/load templates
    audio_templates_path="audio",                # Path to audio samples
    compare_method=CompareMethod.COSINE_SIMILARITY,  # Comparison method
    silence_threshold=0.5,                       # Threshold for silence detection (0-1)
    silence_phoneme="silence"                    # Name of your silence phoneme folder
)
```

## How It Works

1. **Template Building**: On first run, the library analyzes your audio samples and creates MFCC (Mel-frequency cepstral coefficients) templates, saved to `phonemes.json` (or whatever phoneme_templates_path you choose)
2. **Audio Processing**: Input audio is analyzed in segments using the same MFCC extraction
3. **Phoneme Matching**: Each segment is compared against all phoneme templates
4. **Target Calculation**: Returns confidence scores (0-1) for each phoneme per segment

## Credits

This is a Python implementation of [uLipSync](https://github.com/hecomi/uLipSync) by Hecomi. 

## License

MIT License - see [LICENSE](LICENSE) file for details.