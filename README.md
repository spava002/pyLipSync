# pylipsync

A Python implementation of [Hecomi's uLipSync](https://github.com/hecomi/uLipSync) for audio-based lip sync analysis. This library analyzes audio and determines phoneme targets for lip synchronization in real-time applications.

## Installation

### Install from PyPI

```bash
pip install pylipsync
```

### Install from Local Clone

Alternatively, clone the repository and install:

```bash
git clone https://github.com/spava002/pyLipSync.git
cd pyLipSync
pip install -e .
```

## Quick Start

The library comes with built-in audio templates for common phonemes, so you can start using it immediately:

```python
import librosa as lb
from pylipsync import LipSync, CompareMethod

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
    most_prominent_phoneme = segment.most_prominent_phoneme() if not segment.is_silence() else None
    print(f"({segment.start_time:.4f}-{segment.end_time:.4f})s | Most Prominent Phoneme: {most_prominent_phoneme}")
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

## How It Works

1. **Template Loading**: The library loads pre-computed MFCC templates from `data/phonemes.json`
2. **Audio Processing**: Input audio is processed in overlapping windows using MFCC extraction
3. **Phoneme Matching**: Each segment is compared against all phoneme templates using the selected comparison method
4. **Target Calculation**: Returns normalized confidence scores (0-1) for each phoneme per segment
5. **Silence Detection**: Segments below the silence threshold have all phoneme targets set to 0

## Credits

This is a Python implementation of [uLipSync](https://github.com/hecomi/uLipSync) by Hecomi.

## License

MIT License - see [LICENSE](LICENSE) file for details.