#!/usr/bin/env python3
# coding: utf-8
import librosa

class Audio:

    """
    This is the class for working with audio files in the audio low-level descriptors toolbox.
    @author: joachimpoutaraud
    """
    def __init__(self, audio_file, frame_size=2048, hop_size=512, sr=None):

        """
        Args:
            audio_file (str): Path to the audio file.
            frame_size (int, optional): Length of analysis frame (in samples).
            hop_size (int, optional): Number of samples between the successive frames.
            sr (int, optional): Audio sampling rate.
        """

        self.filename = audio_file
        original_sr = librosa.get_samplerate(self.filename)
        self.audio_file, self.sr = librosa.load(audio_file, sr=original_sr) # loading signal with librosa
        if sr is not None:
            self.audio_file, self.sr = librosa.load(audio_file, sr=sr)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.shape = self.audio_file.shape

    from lowleveldescriptors.speech import aaa as aaa
    from lowleveldescriptors.speech import rms as rms
    from lowleveldescriptors.speech import zcr as zcr
    from lowleveldescriptors.speech import pda as pda
    from lowleveldescriptors.speech import hnr as hnr

    
    from lowleveldescriptors.sound import spectral_centroid as spectral_centroid
    from lowleveldescriptors.sound import spectral_bandwidth as spectral_bandwidth
    from lowleveldescriptors.sound import spectrograms as spectrograms

    def __repr__(self):
        return f"Audio('{self.filename}')"
