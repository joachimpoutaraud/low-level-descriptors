#!/usr/bin/env python3
# coding: utf-8

"""
This module encapsulates multiple audio feature extractors into a streamlined and modular implementation.
Class main properties:

- Audio file (file path)
- Frame size (window)
- Hop size (overlap length)
- Sampling rate
- Shape

Features to extract:

- Amplitude envelope
- Average absolute amplitude (AAA)
- Root mean square (RMS)
- Spectral centroid
- Spectral bandwidth
- Zero crossing rate (ZCR)
- Spectrograms (Linear, Log-frequency, Mel)

@author: joachimpoutaraud

"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class AudioFeature():

    def __init__(self, audio_file, frame_size=2048, hop_size=512, sr=44100):

        # loading signal with librosa

        self.audio_file, self.sr = librosa.load(audio_file, sr=sr)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.shape = self.audio_file.shape

    def amplitude_envelope(self, show=False):

        """
        Compute maximum amplitude value for each frame from the audio samples.
        (To visualize amplitude envelope set argument show to True)

        """

        amplitude_envelope = np.zeros((int(self.shape[0]/self.hop_size)+1,))

        # compute maximum amplitude value for each frame
        for i, value in enumerate(range(0, self.shape[0], self.hop_size)):
            current_frame_amplitude_envelope = max(self.audio_file[value:value+self.frame_size])
            amplitude_envelope[i] = current_frame_amplitude_envelope

        if show == True:

            frames = range(0, amplitude_envelope.shape[0])
            frames_to_time = [(i * self.hop_size / self.sr) for i in frames]

            plt.figure(figsize=(15, 4))
            librosa.display.waveplot(self.audio_file, alpha=0.3, sr=self.sr)
            plt.title("Amplitude envelope")
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.plot(frames_to_time, amplitude_envelope, color="r", label='Amplitude envelope')
            plt.ylim(-1, 1)
            plt.legend()

            return plt.show()

        else:
            return amplitude_envelope
        
        
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

            frames = range(0, aaa.shape[0])
            frames_to_time = [(i * self.hop_size / self.sr) for i in frames]

            plt.figure(figsize=(16, 4))
            librosa.display.waveplot(self.audio_file, alpha=0.3, sr=self.sr)
            plt.title("Average absolute amplitude")
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Magnitude')
            plt.plot(frames_to_time, aaa, color="g", label='AAA')
            plt.ylim(-1, 1)
            plt.legend()

            return plt.show()


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

            frames = range(0, rms.shape[0])
            frames_to_time = [(i * self.hop_size / self.sr) for i in frames]

            plt.figure(figsize=(16, 4))
            librosa.display.waveplot(self.audio_file, alpha=0.3, sr=self.sr)
            plt.title("Root-mean-square energy")
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Magnitude')
            plt.plot(frames_to_time, rms, color="g", label='RMS')
            plt.ylim(-1, 1)
            plt.legend()
            return plt.show()

        else:
            return rms

    def spectral_centroid(self, show=False):

        """
        Compute the spectral centroid.
        Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins,
        from which the mean (centroid) is extracted per frame.

        More precisely, the centroid at frame t is defined as: centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])
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
            frames_to_time = [(i * self.hop_size / self.sr) for i in frames]

            plt.figure(figsize=(16, 4))
            plt.title("Spectral centroid")
            plt.plot(frames_to_time, centroid, color="r", label='Spectral centroid')
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
            frames_to_time = [(i * self.hop_size / self.sr) for i in frames]

            plt.figure(figsize=(16, 4))
            plt.title("Spectral bandwidth")
            plt.plot(frames_to_time, bandwidth, color="b", label='Spectral bandwidth')
            plt.plot(frames_to_time, centroid, color="r", label='Spectral centroid', alpha=0.5)
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (Seconds)')
            plt.legend()
            return plt.show()

        else:
            return bandwidth


    def zcr(self, show=False):

        """
        Compute zero crossing rate with librosa.
        (To visualize zero crossing rate set argument show to True)

        """
        # compute zero crossing rate with librosa
        zcr = librosa.feature.zero_crossing_rate(self.audio_file, frame_length=self.frame_size, hop_length=self.hop_size)[0]

        if show:

            frames = range(0, zcr.shape[0])
            frames_to_time = [(i * self.hop_size / self.sr) for i in frames]
            
            plt.figure(figsize=(15, 4))
            plt.title("Zero crossing rate (ZCR)")
            plt.plot(frames_to_time, zcr, color="r", label='ZCR')
            plt.ylabel('Magnitude')
            plt.xlabel('Time (Seconds)')
            plt.legend()
            plt.ylim(-0.5, 0.5)

            return plt.show()

        else:
            return zcr

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
