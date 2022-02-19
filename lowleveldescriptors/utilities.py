import numpy as np

def framing(signal, frame_size, hop_size):

    # Get the number of samples
    num_samples = np.asarray(signal).shape[0]
    # Compute the overlap between each frame
    frame_overlap = frame_size - hop_size
    # Compute the expected number of frames
    num_frames = 1 + (num_samples - frame_size) // hop_size
    # Compute the rest of samples
    rest_samples = (num_samples - frame_overlap) % (frame_size - frame_overlap)

    if rest_samples != 0:

        # Pad the signal with the rest of samples converted to zeros
        pad_num_samples = hop_size - rest_samples
        zeros = np.zeros((pad_num_samples))
        pad_signal = np.append(signal, zeros)
        # Add another frame for the rest of samples padded
        num_frames += 1

    else:
        pad_signal = signal

    # Create a new array of given shape, filled with zeros
    frames = np.zeros([num_frames,frame_size])

    # Segment the segments with the frame overlap and apply a hanning window
    for i in range(frames.shape[0]):
        frames[i] = pad_signal[i*hop_size:i*hop_size+frame_size] * np.hanning(frame_size)

    return frames


def frames_to_time(frames, hop_size, sr):

    times = np.zeros((np.asarray(frames).shape[0]))
    for i, value in enumerate(frames):
        times[i] = i * hop_size / sr

    return times



