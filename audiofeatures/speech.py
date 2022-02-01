
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import audiofeatures
from audiofeatures.core import frames_to_time
from audiofeatures.util import framing

"""
This module encapsulates multiple audio feature extractors into a streamlined and modular implementation.
Features to extract:

- Average absolute amplitude (AAA)
- Root mean square (RMS)
- Zero crossing rate (ZCR)
"""

def aaa(self, show=False):

    """
    Compute average absolute amplitude (AAA) value for each frame from the audio samples.
    (To visualize average absolute amplitude set argument show to True)

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
    Compute root-mean-square (RMS) value for each frame from the audio samples.
    (To visualize root mean square set argument show to True)

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


def zcr(self, show=False):

    """
    Compute the zero-crossing rate of an audio time series.
    """

    # Slice the data array into (overlapping) frames.
    frames = framing(self.audio_file, self.frame_size, self.hop_size)
    zcr = np.zeros((frames.shape[0]))

    for index, frame in enumerate(frames):
        #To avoid DC bias, usually we need to perform mean subtraction on each frame
        frame = frame-np.mean(frame)
        zcr[index] = sum(frame[0:-1] * frame[1::]<=0) / self.sr

    if show:

        times = frames_to_time(zcr, self.hop_size, self.sr)

        plt.figure(figsize=(15, 4))
        plt.title("Zero crossing rate (ZCR)")
        plt.plot(times, zcr, color="r", label='ZCR')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (Seconds)')
        plt.legend()

        return plt.show()

    else:
        return zcr
