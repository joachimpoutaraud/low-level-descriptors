
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from lowleveldescriptors.utilities import framing, frames_to_time

"""
This module encapsulates multiple audio low-level descriptors into a streamlined and modular implementation.
Low-level descriptors to extract:

- Average absolute amplitude (AAA)
- Root mean square (RMS)
- Zero crossing rate (ZCR)
- Pitch detection algorithm (PDA)
"""

def aaa(self, show=False):

    """
    Compute average absolute amplitude (AAA) value for each frame from the audio samples.

    Args:
        show (bool, optional): Set argument to True to visualize the average absolute amplitude.

    Returns:
        Average absolute amplitude.
    """

    # Pad with the reflection of the signal so that the frames are centered
    # Padding is achieved by mirroring on the first and last values of the signal with frame_length // 2 samples
    signal = np.pad(self.audio_file, int(self.frame_size // 2), mode='reflect')

    aaa = np.zeros((int(self.shape[0]/self.hop_size)+1,))

    for i, value in enumerate(range(0, self.shape[0], self.hop_size)):

        aaa_formula = 1 / self.frame_size * np.sum(np.abs(signal[value:value+self.frame_size]))
        aaa[i] = aaa_formula

    if show:

        times = frames_to_time(aaa, self.hop_size, self.sr)

        plt.figure(figsize=(16, 4))
        librosa.display.waveplot(self.audio_file, alpha=0.3, sr=self.sr)
        plt.title("Average absolute amplitude (AAA)")
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Magnitude')
        plt.plot(times, aaa, color="g", label='AAA')
        plt.ylim(-1, 1)
        plt.legend()

        return plt.show()

    else:
        return aaa


def rms(self, show=False):

    """
    Compute root mean square (RMS) value for each frame from the audio samples.
    
    Args:
        show (bool, optional): Set argument to True to visualize the root mean square.

    Returns:
        Root mean square.
    """

    # Pad with the reflection of the signal so that the frames are centered
    # Padding is achieved by mirroring on the first and last values of the signal with frame_length // 2 samples
    signal = np.pad(self.audio_file, int(self.frame_size // 2), mode='reflect')

    rms = np.zeros((int(self.shape[0]/self.hop_size)+1,))

    for i, value in enumerate(range(0, self.shape[0], self.hop_size)):

        rms_formula = np.sqrt(1 / self.frame_size * np.sum(signal[value:value+self.frame_size]**2))
        rms[i] = rms_formula

    if show:

        times = frames_to_time(rms, self.hop_size, self.sr)

        plt.figure(figsize=(16, 4))
        librosa.display.waveplot(self.audio_file, alpha=0.3, sr=self.sr)
        plt.title("Root mean square (RMS)")
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Magnitude')
        plt.plot(times, rms, color="g", label='RMS')
        plt.ylim(-1, 1)
        plt.legend()
        return plt.show()

    else:
        return rms


def zcr(self, threshold=0, show=False):

    """
    Compute the zero crossing rate of an audio time series.
    
    Args:
        threshold (int, optional): Set the threshold of the crossing rate (default to zero).
        show (bool, optional): Set argument to True to visualize the zero crossing rate.

    Returns:
        Zero crossing rate.
    """

    # Pad with the edge of the signal so that the frames are centered
    signal = np.pad(self.audio_file, int(self.frame_size // 2), mode='edge')

    # Slice the data array into (overlapping) frames.
    frames = framing(signal, self.frame_size, self.hop_size)
    zcr = np.zeros((frames.shape[0]))

    for index, frame in enumerate(frames):
        # Compute zero crossings on each frame
        zero_crossings = frame[:-1] * frame[1:] < threshold
        # Sum zero crossings and normalize with the frame size
        zcr[index] = sum(zero_crossings) / self.frame_size

    if show:

        times = frames_to_time(zcr, self.hop_size, self.sr)

        plt.figure(figsize=(15, 4))
        plt.title("Zero crossing rate (ZCR)")
        plt.plot(times, zcr, color="r", label='ZCR')
        plt.ylabel('Crossing rate')
        plt.xlabel('Time (Seconds)')
        plt.legend()

        return plt.show()

    else:
        return zcr

def pda(self, threshold=0, fmin=0, fmax=11025):

    """
    Pitch detection algorithm (PDA) to estimate the pitch of an audio time series using autocorrelation.
    
    Args:
        threshold (int, optional): Set the threshold of the maximum intensity value between 0 nad 1 (default to 0).
        fmin (int, optional): Set the minimum frequency to estimate the pitch of the audio time series (default to 0).
        fmax (int, optional): Set the maximum frequency to estimate the pitch of the audio time series (default to 11025).

    Returns:
        Fundamental frequency of the selected audio file (or frame) using autocorrelation.
    """
    
    # Return maximum absolute value of the frame array
    frame = self.audio_file.astype(float)
    frame -= frame.mean()
    amax = np.amax(np.abs(frame)) 
    # Normalize the frame array between -1 and 1
    if amax > 0:
        frame /= amax
    else:
        return 0

    # Calculate autocorrelation using numpy correlate
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[corr.shape[0]//2:] # keep only the positive part

    # Find the first minimum peak indice
    diff_corr = np.diff(corr) # calculate autocorrelation's discrete difference
    rmin = np.where(diff_corr > 0)[0] # returns indices of the positive autocorrelation's discrete difference values
    if rmin.shape[0] > 0:
        first_peak = rmin[0]
    else:
        return 0

    # Find the fundamental frequency
    next_peak = np.argmax(corr[first_peak:]) + first_peak # returns the indice of the next peak
    rmax = corr[next_peak]/corr[0] # normalize maximum intensity value
    f0 = self.sr / next_peak # get the fundamental frequency

    if rmax > threshold and f0 >= fmin and f0 <= fmax:
        return f0
    else:
        return 0
