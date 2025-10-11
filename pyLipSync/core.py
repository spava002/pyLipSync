import os, json
import librosa as lb
import numpy as np
from collections import defaultdict
from math import log10
from .algorithms import *
from .types import Phoneme, PhonemeSegment, LipSyncInfo
from .comparison import CompareMethod, calculate_similarity_score

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

class LipSync:
    AUDIO_EXTENSIONS = ("wav", "mp3", "ogg", "flac", "m4a", "wma", "aac", "aiff", "au", "raw", "pcm")
    TARGET_SAMPLE_RATE = 16000
    RANGE_HZ = 500
    MIN_VOLUME = -2.5
    MAX_VOLUME = -1.5

    def __init__(
        self, 
        phoneme_templates_path: str = "phonemes.json",
        audio_templates_path: str = "audio",
        compare_method: CompareMethod = CompareMethod.L2_NORM,
    ):
        """
        A pipeline for analyzing audio and determining the best phoneme for a given audio chunk.

        Args:
            phoneme_templates_path: Path or file name to the phoneme templates file located within the module directory
            audio_templates_path: Path or file name to the audio templates directory located within the module directory
            compare_method: Method to use for comparing MFCCs (L1Norm, L2Norm, or CosineSimilarity)
        """
        self.phoneme_templates_path = os.path.join(MODULE_DIR, phoneme_templates_path)
        self.audio_templates_path = os.path.join(MODULE_DIR, audio_templates_path)

        self.phoneme_templates = self._get_phoneme_templates()
        self.means, self.std_devs = self._calculate_means_and_stds(self.phoneme_templates)
        self.compare_method = compare_method

    def _get_phoneme_templates(self) -> dict[str, list]:
        """Get the phoneme templates from the file or create them if they don't exist.
        
        Returns:
            A dictionary of phonemes and their vectors
        """
        try:
            phoneme_templates = self._load_phoneme_templates()
            print(f"Phoneme templates loaded from {self.phoneme_templates_path}")
        except FileNotFoundError:
            print(f"Phoneme templates file not found at {self.phoneme_templates_path}, building templates...")
            phoneme_templates = self._create_phoneme_templates()
            print(f"Phoneme templates built and saved to {self.phoneme_templates_path}")
        return phoneme_templates

    def _load_phoneme_templates(self) -> dict[str, list]:
        """Load the phoneme templates from the file.
        
        Returns:
            A dictionary of phonemes and their vectors
        """
        with open(self.phoneme_templates_path) as f:
            return json.load(f)

    def _create_phoneme_templates(self) -> dict[str, list]:
        """Create phoneme templates from the audio templates.
        
        Returns:
            A dictionary of phonemes and their vectors
        """
        phoneme_templates = defaultdict(list)

        phoneme_audio_folders = [
            os.path.join(self.audio_templates_path, folder)
            for folder in os.listdir(self.audio_templates_path) 
            if os.path.isdir(os.path.join(self.audio_templates_path, folder))
        ]

        if not phoneme_audio_folders:
            raise FileNotFoundError(f"No folders found within {self.audio_templates_path}!")

        for phoneme_audio_folder in phoneme_audio_folders:
            phoneme = os.path.basename(phoneme_audio_folder)
            
            for file in os.listdir(phoneme_audio_folder):
                # Skip if not an audio file
                if not file.endswith(self.AUDIO_EXTENSIONS):
                    continue

                # Load audio file
                audio_file_path = os.path.join(phoneme_audio_folder, file)
                audio_data, sample_rate = lb.load(audio_file_path, sr=None)

                # Calculate MFCC and add to phoneme dictionary
                mfcc = self._process_audio(audio_data, sample_rate, calculate_scores=False).mfcc
                phoneme_templates[phoneme].append(mfcc)

        if not phoneme_templates:
            raise FileNotFoundError(f"Could not create phoneme templates. No audio files were found!")

        with open(self.phoneme_templates_path, "w") as f:
            json.dump(phoneme_templates, f, indent=4)
        
        return phoneme_templates

    def _calculate_means_and_stds(self, phonemes_templates: dict[str, list]) -> tuple[list, list]:
        """Calculate means and standard deviations across all phoneme vectors.
        
        Args:
            phonemes_templates: A dictionary of phonemes and their vectors

        Returns:
            A tuple of means and standard deviations, where the first element is the means and the second element is the standard deviations.
        """
        # Flatten all vectors into a single list
        all_vectors = []
        for vectors in phonemes_templates.values():
            for vector in vectors:
                all_vectors.append(vector)
    
        # Convert to numpy array for easier calculation
        all_vectors = np.array(all_vectors)
        
        # Calculate means and standard deviations for each dimension
        means = np.mean(all_vectors, axis=0)
        std_devs = np.std(all_vectors, axis=0)
        
        # Avoid division by zero - set very small std_devs to 1.0
        std_devs = np.where(std_devs < 1e-10, 1.0, std_devs)
        
        return means.tolist(), std_devs.tolist()

    def _calculate_scores(self, mfcc: np.ndarray, volume: float) -> LipSyncInfo:
        """Calculate scores for each phoneme template and normalize them to sum to 1.
        
        Args:
        mfcc: np.ndarray
            The MFCC to calculate scores for

        Returns:
            A list of Phoneme objects, one per phoneme
        """
        means = np.array(self.means)
        std_devs = np.array(self.std_devs)
        
        # Standardize the input MFCC
        normalized_mfcc = (mfcc - means) / std_devs
        
        phonemes = []
        # Calculate score for each phoneme
        for phoneme, templates in self.phoneme_templates.items():
            phoneme_score = 0.0
            
            for template in templates:
                template_array = np.array(template)
                
                # Standardize the template MFCC
                normalized_template = (template_array - means) / std_devs
                
                # Calculate score using the comparison method
                score = calculate_similarity_score(normalized_mfcc, normalized_template, self.compare_method)
                phoneme_score += score

            phonemes.append(Phoneme(phoneme, phoneme_score))

        normalized_volume = self._normalize_volume(volume)

        normalized_phonemes = self._normalize_phonemes(phonemes)
        phonemes = self._apply_volume_weighting(normalized_phonemes, normalized_volume)
        return LipSyncInfo(mfcc.tolist(), volume, normalized_volume, phonemes)

    def _normalize_phonemes(self, phonemes: list[Phoneme]) -> list[Phoneme]:
        """Normalizes phoneme target values to sum of 1.
        
        Args:
            phonemes: A list of Phoneme objects
        """
        total_target = sum(phoneme.target for phoneme in phonemes)
        for phoneme in phonemes:
            phoneme.target = phoneme.target / total_target if total_target > 0 else 0.0
        return phonemes

    def _normalize_volume(self, volume: float) -> float:
        norm_volume = log10(volume)
        norm_volume = (norm_volume - self.MIN_VOLUME) / max(self.MAX_VOLUME - self.MIN_VOLUME, 1e-4)
        norm_volume = min(max(norm_volume, 0), 1)
        return norm_volume

    def _apply_volume_weighting(self, phonemes: list[Phoneme], normalized_volume: float) -> list[Phoneme]:
        for phoneme in phonemes:
            phoneme.target *= normalized_volume
        return phonemes

    def _process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        mel_channels: int = 26,
        mfcc_num: int = 12,
        calculate_scores: bool = True
    ) -> LipSyncInfo:
        """
        Process the audio data and return the MFCC and scores.
        
        Args:
            audio_data: The audio data to process
            sample_rate: The sample rate of the audio data
            mel_channels: The number of Mel Filter Bank channels
            mfcc_num: The number of MFCC coefficients
            calculate_scores: Whether to calculate the scores

        Returns:
            A LipSyncInfo object, containing the MFCC and scores.
        """

        volume = rms_volume(audio_data)

        cutoff = self.TARGET_SAMPLE_RATE / 2
        filtered = low_pass_filter(audio_data, sample_rate, cutoff, self.RANGE_HZ)
        downsampled = downsample(filtered, sample_rate, self.TARGET_SAMPLE_RATE)
        emphasized = pre_emphasis(downsampled, 0.97)
        windowed = hamming_window(emphasized)
        normalized = normalize_array(windowed, 1.0)
        padded = zero_padding(normalized)
        spectrum = fft_magnitude(padded)
        mel_spectrum = mel_filter_bank(spectrum, self.TARGET_SAMPLE_RATE, mel_channels)
        mel_db = power_to_db(mel_spectrum)
        mel_cepstrum = dct(mel_db)
        mfcc = mel_cepstrum[1: mfcc_num + 1]

        return self._calculate_scores(mfcc, volume) if calculate_scores else LipSyncInfo(mfcc.tolist())

    def process_audio_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        window_size_ms: float = 64.0,
        fps: int = 60
    ) -> list[PhonemeSegment]:
        """
        Process an entire audio file in segments and return phoneme predictions for each window.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate of the audio data
            window_size_ms: Window size in milliseconds
            fps: Frames per second
        
        Returns:
            List of PhonemeSegment objects, one per window.
        """
        
        downsampled_audio = downsample(audio_data, sample_rate, self.TARGET_SAMPLE_RATE)
        
        # Calculate window and hop sizes
        window_size = int((window_size_ms / 1000) * self.TARGET_SAMPLE_RATE)
        hop_size = self.TARGET_SAMPLE_RATE // fps
        sample_rate_ratio = sample_rate / self.TARGET_SAMPLE_RATE
        
        segments = []
        for i in range(0, len(downsampled_audio) - window_size + 1, hop_size):
            # Calculate window for analysis (with overlap)
            downsampled_chunk = downsampled_audio[i: i + window_size]
            
            # But only store the non-overlapping hop_size portion in original sample rate
            original_hop_start = int(i * sample_rate_ratio)
            original_hop_end = int((i + hop_size) * sample_rate_ratio)
            original_hop_chunk = audio_data[original_hop_start: original_hop_end]
            
            # Process with full window, store only hop
            lipsync_info = self._process_audio(downsampled_chunk, self.TARGET_SAMPLE_RATE)
            segments.append(
                PhonemeSegment(lipsync_info.volume, lipsync_info.normalized_volume, original_hop_chunk, lipsync_info.phonemes
            ))
        
        return segments