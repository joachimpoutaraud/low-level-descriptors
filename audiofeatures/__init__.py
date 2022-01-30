
#!/usr/bin/env python3
# coding: utf-8
import librosa

class Audio:

    """
    This is the class for working with audio files in the Audio Features Toolbox.
    @author: joachimpoutaraud
    """
    def __init__(self, audio_file, frame_size=2048, hop_size=512, sr=44100):

        """
        Args:
            audio_file (str): Path to the audio file.
            frame_size (int): Length of analysis frame (in samples).
            hop_size (int): Number of samples between the successive frames.
            sr (int): Audio sampling rate.
        """

        self.filename = audio_file
        self.audio_file, self.sr = librosa.load(audio_file, sr=sr) # loading signal with librosa
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.shape = self.audio_file.shape

    from audiofeatures.features import aaa as aaa
    from audiofeatures.features import rms as rms
    from audiofeatures.features import zcr as zcr
    from audiofeatures.features import spectral_centroid as spectral_centroid
    from audiofeatures.features import spectral_bandwidth as spectral_bandwidth
    from audiofeatures.features import spectrograms as spectrograms

    def __repr__(self):
        return f"Audio('{self.filename}')"
