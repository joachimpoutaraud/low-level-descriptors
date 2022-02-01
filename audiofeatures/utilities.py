import numpy as np

def framing(signal, frame_length, hop_length):

    # Get the number of samples
    num_samples = np.asarray(signal).shape[0]
    # Compute the overlap between each frame
    frame_overlap = frame_length - hop_length
    # Compute the expected number of frames
    num_frames = 1 + (num_samples - frame_length) // hop_length
    # Compute the rest of samples
    rest_samples = (num_samples - frame_overlap) % (frame_length - frame_overlap)

    if rest_samples != 0:

        # Pad the signal with the rest of samples converted to zeros
        pad_num_samples = hop_length - rest_samples
        zeros = np.zeros((pad_num_samples))
        pad_signal = np.append(signal, zeros)
        # Add another frame for the rest of samples padded
        num_frames += 1

    else:
        pad_signal = signal

    # Create a new array of given shape, filled with zeros
    frames = np.zeros([num_frames,frame_length])

    # Segment the segments with the frame overlap and apply a hanning window
    for i in range(frames.shape[0]):
        frames[i] = pad_signal[i*hop_length:i*hop_length+frame_length] * np.hanning(frame_length)

    return frames


def frames_to_time(frames, hop_size, sr):

    times = np.zeros((np.asarray(frames).shape[0]))
    for i, value in enumerate(frames):
        times[i] = i * hop_size / sr

    return times



