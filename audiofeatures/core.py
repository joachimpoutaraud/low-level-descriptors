import numpy as np

def frames_to_time(frames, hop_size, sr):

    frames = np.asarray(frames)
    times = np.zeros((frames.shape[0],))

    for i in frames:

        times[i] = i * hop_size / sr

    return times
