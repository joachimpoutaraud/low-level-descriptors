
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from lowleveldescriptors.utilities import frames_to_time

"""
This module encapsulates multiple audio feature extractors into a streamlined and modular implementation.
Features to extract:

- Spectral centroid
- Spectral bandwidth
- Spectrograms (Linear, Log-frequency, Mel)
"""

def spectral_centroid(self, show=False):

    """
    Compute the spectral centroid.
    Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins,
    from which the mean (centroid) is extracted per frame.

    More precisely, the centroid at frame t is defined as: centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])

    Args:
        show (bool, optional): Set argument to True to visualize the spectral centroid.

    Returns:
        Spectral centroid.
    """

    # Pad with the reflection of the signal so that the frames are centered
    # Padding is achieved by mirroring on the first and last values of the signal with frame_length // 2 samples
    signal = np.pad(self.audio_file, int(self.frame_size // 2), mode='reflect')

    centroid = np.zeros((int(self.shape[0]/self.hop_size)+1,))

    for i, value in enumerate(range(0, self.shape[0], self.hop_size)):

        cent = signal[value:value+self.frame_size]

        # Compute the discrete Fourier Transform (DFT) with the efficient Fast Fourier Transform (FFT)
        magnitudes = np.abs(np.fft.fft(cent)) # magnitude of absolute (real) frequency values

        # Compute only the positive half of the DFT (i.e 1 + first half)
        mag = magnitudes[:int(1 + len(magnitudes) // 2)]

        # Compute the center frequencies of each bin
        freq = np.linspace(0, self.sr/2, int(1 + len(cent) // 2))

        # Return weighted mean of the frequencies present in the signal
        normalize_mag = np.nan_to_num(mag / np.sum(mag))
        np.seterr(invalid='ignore') # Hide true_divide warning
        centroid[i] = np.sum(freq * normalize_mag)

    if show:

        frames = range(0, centroid.shape[0])
        times = frames_to_time(frames, self.hop_size, self.sr)

        plt.figure(figsize=(16, 4))
        plt.title("Spectral centroid")
        plt.plot(times, centroid, color="r", label='Spectral centroid')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (Seconds)')
        plt.legend()
        return plt.show()

    else:
        return centroid


def spectral_bandwidth(self, p=2, show=False):

    """
    Compute p’th-order spectral bandwidth.
    The spectral bandwidth 1 at frame t is computed by: (sum_k S[k, t] * (freq[k, t] - centroid[t])**p)**(1/p)

    Args:
        p (int, optional): Power to raise deviation from spectral centroid.
        show (bool, optional): Set argument to True to visualize the spectral bandwidth.

    Returns:
        Spectral bandwidth.
    """

    # Pad with the reflection of the signal so that the frames are centered
    # Padding is achieved by mirroring on the first and last values of the signal with frame_length // 2 samples
    signal = np.pad(self.audio_file, int(self.frame_size // 2), mode='reflect')

    centroid = np.zeros((int(self.shape[0]/self.hop_size)+1,))
    bandwidth = np.zeros((int(self.shape[0]/self.hop_size)+1,))

    for i, value in enumerate(range(0, self.shape[0], self.hop_size)):

        frame = signal[value:value+self.frame_size]

        # Compute the discrete Fourier Transform (DFT) with the efficient Fast Fourier Transform (FFT)
        magnitudes = np.abs(np.fft.fft(frame)) # magnitude of absolute (real) frequency values

        # Compute only the positive half of the DFT (i.e 1 + first half)
        mag = magnitudes[:int(1 + len(magnitudes) // 2)]

        # Compute the center frequencies of each bin
        freq = np.linspace(0, self.sr/2, int(1 + len(frame) // 2))

        # Return weighted mean of the frequencies present in the signal
        normalize_mag = np.nan_to_num(mag / np.sum(mag))
        np.seterr(invalid='ignore') # Hide true_divide warning
        cent = np.sum(freq * normalize_mag)
        centroid[i] = cent

        spectral_bandwidth = np.sum(normalize_mag * abs(freq - cent) ** p) ** (1.0/p)
        bandwidth[i] = spectral_bandwidth

    if show:

        frames = range(0, bandwidth.shape[0])
        times = frames_to_time(frames, self.hop_size, self.sr)

        plt.figure(figsize=(16, 4))
        plt.title("Spectral bandwidth")
        plt.plot(times, bandwidth, color="b", label='Spectral bandwidth')
        plt.plot(times, centroid, color="r", label='Spectral centroid', alpha=0.5)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (Seconds)')
        plt.legend()
        return plt.show()

    else:
        return bandwidth


def spectrograms(self, showLinear=False, showLog=False, showMel=False):

    """
    Compute conventional spectrograms.
    (To visualize linear-frequency spectrogram set argument showLinear to True)
    (To visualize log-frequency spectrogram set argument showLog to True)
    (To visualize mel spectrogram set argument showMel to True)

    """

    # extract short time fourier transform with librosa
    stft = librosa.stft(self.audio_file, n_fft=self.frame_size, hop_length=self.hop_size)

    # compute spectrogram and move amplitude to logarithmic scale
    spec_log_amplitude = librosa.power_to_db(np.abs(stft) ** 2)

    # extracting mel spectrogram with librosa
    mel_spectrogram = librosa.feature.melspectrogram(self.audio_file,
                                                     n_fft=self.frame_size, hop_length=self.hop_size, n_mels=90)

    # compute mel spectrogram and move amplitude to logarithmic scale
    mel_spectrogram_log = librosa.power_to_db(mel_spectrogram)

    if showLinear == True:
        # visualize spectrogram with librosa function
        plt.figure(figsize=(25, 10))
        plt.title("Linear-frequency spectrogram")
        librosa.display.specshow(spec_log_amplitude, hop_length=self.hop_size, x_axis="time", y_axis="linear")
        plt.colorbar(format="%+2.f")

        return plt.show()

    elif showLog == True:
        # visualize spectrogram with librosa function
        plt.figure(figsize=(25, 10))
        plt.title("Logarithmic-frequency spectrogram")
        librosa.display.specshow(spec_log_amplitude, hop_length=self.hop_size, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.f")

        return plt.show()

    elif showMel == True:
        # visualize mel spectrogram with librosa function
        plt.figure(figsize=(25,10))
        plt.title("Mel spectrogram")
        librosa.display.specshow(mel_spectrogram_log, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.f")

        return plt.show()

    else:
        return np.array(spec_log_amplitude)
