import numpy as np
import scipy.signal
import librosa as lb

def rms_volume(audio_data: np.ndarray) -> float:
    """Calculates the RMS of the audio data."""
    return np.sqrt(np.mean(audio_data ** 2))

def low_pass_filter(audio_data: np.ndarray, sample_rate: int, cutoff: float, range_hz: float) -> np.ndarray:
    """Applies a low pass filter to the audio data using scipy."""
    # Use the exact same calculations
    cutoff_norm = (cutoff - range_hz) / sample_rate
    range_norm = range_hz / sample_rate
    
    # Calculate filter length
    n = int(np.round(3.1 / range_norm))
    if (n + 1) % 2 == 0:
        n += 1
    
    # Create filter coefficients using the exact same sinc calculation
    b = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = i - (n - 1) / 2.0
        ang = 2.0 * np.pi * cutoff_norm * x
        if abs(ang) < 1e-10:
            b[i] = 2.0 * cutoff_norm
        else:
            b[i] = 2.0 * cutoff_norm * np.sin(ang) / ang
    
    # Apply the filter using scipy
    filtered = scipy.signal.lfilter(b, 1.0, audio_data)
    
    return filtered.astype(np.float32)

def downsample(audio_data: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Downsamples the audio data to the target sample rate using librosa."""
    if sample_rate <= target_sample_rate:
        return audio_data.copy()
    
    resampled = lb.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
    
    return resampled.astype(np.float32)

def pre_emphasis(data: np.ndarray, p: float = 0.97) -> np.ndarray:
    """Applies pre-emphasis filter using simple calculation."""
    # Simple vectorized calculation
    emphasized = data.copy().astype(np.float32)
    emphasized[1:] = data[1:] - p * data[:-1]
    return emphasized

def hamming_window(array: np.ndarray) -> np.ndarray:
    """Applies Hamming window using numpy's built-in function."""
    # Use numpy's built-in Hamming window
    window = np.hamming(len(array)).astype(np.float32)
    return (array * window).astype(np.float32)

def normalize_array(array: np.ndarray, value: float = 1.0) -> np.ndarray:
    """Normalizes the array to the specified maximum value using numpy."""
    # Find maximum absolute value
    max_val = np.max(np.abs(array))
    
    # Check if max_val is essentially zero
    if max_val < np.finfo(float).eps:
        return array.astype(np.float32)
    
    # Scale the array (vectorized operation - much faster than loops)
    return (array * (value / max_val)).astype(np.float32)

def zero_padding(data: np.ndarray) -> np.ndarray:
    """Applies zero padding using numpy."""
    n = len(data)
    # Create array of zeros with double the size
    padded = np.zeros(n * 2, dtype=np.float32)
    # Place original data in the center
    padded[n//2:n//2 + n] = data
    return padded

def fft_magnitude(data: np.ndarray) -> np.ndarray:
    """Computes FFT and returns magnitude spectrum using numpy."""
    # Use numpy's optimized FFT implementation
    fft_result = np.fft.fft(data)
    # Return magnitude (absolute value)
    return np.abs(fft_result).astype(np.float32)

def to_mel(hz: float, slaney: bool = False) -> float:
    """Converts Hz to Mel scale."""
    a = 2595.0 if slaney else 1127.0
    return a * np.log(hz / 700.0 + 1.0)

def to_hz(mel: float, slaney: bool = False) -> float:
    """Converts Mel scale to Hz."""
    a = 2595.0 if slaney else 1127.0
    return 700.0 * (np.exp(mel / a) - 1.0)

def mel_filter_bank(spectrum: np.ndarray, sample_rate: int, mel_div: int) -> np.ndarray:
    """Optimized mel filter bank using vectorized numpy operations."""
    f_max = sample_rate / 2
    mel_max = to_mel(f_max)
    n_max = len(spectrum) // 2
    df = f_max / n_max
    d_mel = mel_max / (mel_div + 1)
    
    # Vectorized calculation of mel frequencies
    mel_points = np.arange(mel_div + 2) * d_mel
    f_points = np.array([to_hz(mel) for mel in mel_points])
    
    mel_spectrum = np.zeros(mel_div, dtype=np.float32)
    
    for n in range(mel_div):
        i_begin = int(np.ceil(f_points[n] / df))
        i_center = int(np.round(f_points[n + 1] / df))
        i_end = int(np.floor(f_points[n + 2] / df))
        
        if i_end > i_begin:
            # Vectorized calculation of frequency bins and weights
            indices = np.arange(i_begin + 1, i_end + 1)
            frequencies = df * indices
            
            # Vectorized triangular weight calculation
            left_mask = indices < i_center
            right_mask = indices >= i_center
            
            weights = np.zeros_like(frequencies)
            if np.any(left_mask):
                weights[left_mask] = (frequencies[left_mask] - f_points[n]) / (f_points[n + 1] - f_points[n])
            if np.any(right_mask):
                weights[right_mask] = (f_points[n + 2] - frequencies[right_mask]) / (f_points[n + 2] - f_points[n + 1])
            
            # Normalize weights
            weights /= (f_points[n + 2] - f_points[n]) * 0.5
            
            # Apply weights to spectrum - vectorized sum
            mel_spectrum[n] = np.sum(weights * spectrum[indices])
    
    return mel_spectrum

def power_to_db(array: np.ndarray) -> np.ndarray:
    """Converts power to dB using numpy."""
    return (10.0 * np.log10(np.maximum(array, 1e-10))).astype(np.float32)

def dct(spectrum: np.ndarray) -> np.ndarray:
    """Computes DCT using the exact same method."""
    n = len(spectrum)
    cepstrum = np.zeros(n, dtype=np.float32)
    a = np.pi / n
    
    # Vectorized computation for better performance
    for i in range(n):
        # Create array of j values
        j_vals = np.arange(n)
        # Vectorized angle calculation
        angles = (j_vals + 0.5) * i * a
        # Vectorized cosine and sum
        cepstrum[i] = np.sum(spectrum * np.cos(angles))
    
    return cepstrum