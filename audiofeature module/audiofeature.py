#!/usr/bin/env python3
# coding: utf-8

"""
This module encapsulates multiple audio feature extractors into a streamlined and modular implementation.
Class main properties:
    
- Audio file (file path)
- Frame size (window)
- Hop size (overlap length)
    
Features to extract:
    
- Amplitude envelope
- Root mean square
- Zero crossing rate
- Spectrograms (Linear, Log-frequency, Mel)

@author: joachimpoutaraud

"""
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

class AudioFeature():
    
    def __init__(self, audio_file, frame_size=1024, hop_size=512):
        
        # loading signal with librosa        
        
        self.audio_file, sr = librosa.load(audio_file)
        self.frame_size = frame_size
        self.hop_size = hop_size

    def amplitude_envelope(self, show=False):
        
        """
        Calculate amplitude envelope
        (To visualize amplitude envelope set argument show to True)
        
        """

        amplitude_envelope = []
        
        # calculate maximum amplitude value for each frame
        for i in range (0, len(self.audio_file), self.hop_size):
            current_frame_amplitude_envelope = max(self.audio_file[i:i+self.frame_size])
            amplitude_envelope.append(current_frame_amplitude_envelope)

        if show == True:
            ae = np.array(amplitude_envelope) 

            frames= range(0, ae.size)
            t= librosa.frames_to_time(frames) 

            plt.figure(figsize=(15, 4)) 
            librosa.display.waveplot(self.audio_file, alpha=0.3) 
            plt.title("Amplitude envelope")
            plt.plot(t, ae, color="r")
            plt.ylim(-1, 1) 

            return plt.show()
        
        else:
            return np.array(amplitude_envelope)

    def rms(self, show=False):
        
        """
        Calculate root mean square
        (To visualize root mean square set argument show to True)
        
        """
        rms = []

        for i in range(0, len(self.audio_file), self.hop_size):
            # RMS mathematical formula
            rms_current_frame = np.sqrt(np.sum(self.audio_file[i:i+self.frame_size]**2) / self.frame_size)
            rms.append(rms_current_frame)
            
        if show == True:
            rms1 = np.array(rms)   

            frames= range(0, rms1.size)
            t= librosa.frames_to_time(frames)

            plt.figure(figsize=(15, 4)) 
            librosa.display.waveplot(self.audio_file, alpha=0.3) 
            plt.title("Root-mean-square energy")
            plt.plot(t, rms1, color="g")
            plt.ylim(-1, 1) 

            return plt.show()
        
        else:
            return np.array(rms)
        
    def zcr(self, show=False):
        
        """
        Calculate zero crossing rate
        (To visualize zero crossing rate set argument show to True)
        
        """  
        # calculate zero crossing rate with librosa
        zcr = librosa.feature.zero_crossing_rate(self.audio_file, frame_length=self.frame_size, hop_length=self.hop_size)[0]

        if show == True:
            
            frames= range(0, len(zcr))
            t=librosa.frames_to_time(frames, hop_length=self.hop_size) 
            plt.figure(figsize=(15, 4))
            plt.title("Zero crossing rate")
            plt.plot(t, zcr, color="r")
            plt.ylim(0, 1)
            
            return plt.show()
        
        else:
            return zcr
        
    def spectrograms(self, showLinear=False, showLog=False, showMel=False):
        
        """
        Calculate conventional spectrogram
        (To visualize linear-frequency spectrogram set argument showLinear to True)
        (To visualize log-frequency spectrogram set argument showLog to True)
        (To visualize mel spectrogram set argument showMel to True)
        
        """

        # extract short time fourier transform with librosa
        stft = librosa.stft(self.audio_file, n_fft=self.frame_size, hop_length=self.hop_size)
                
        # calculate spectrogram and move amplitude to logarithmic scale
        spec_log_amplitude = librosa.power_to_db(np.abs(stft) ** 2)
        
        # extracting mel spectrogram with librosa
        mel_spectrogram = librosa.feature.melspectrogram(self.audio_file, 
                                                         n_fft=self.frame_size, hop_length=self.hop_size, n_mels=90)
        
        # calculate mel spectrogram and move amplitude to logarithmic scale
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