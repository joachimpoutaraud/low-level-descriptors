import numpy as np

def frames_to_time(frames, hop_size, sr):

    times = np.zeros((np.asarray(frames).shape[0]))
    for i, value in enumerate(frames):
        times[i] = i * hop_size / sr

    return times
