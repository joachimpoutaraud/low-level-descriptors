# Audio low-level descriptors (LLDs)

This module encapsulates multiple audio low-level descriptors into a streamlined and modular implementation.

## Prerequisites

- librosa
- matplotlib
- numpy

(see requirements.txt. Install with `pip install -r requirements.txt`)

## Importing low-level descriptors

```python
import lowleveldescriptors as lld
descriptors = lld.Audio('path_to_your_audio_file')

# Extract pitch with the pitch detection algorithm (PDA)
descriptors.pda()
```

## Low-level descriptors

*Speech*
- Average Absolute Amplitude (AAA)
- Root mean square (RMS)
- Zero crossing rate (ZCR)
- Pitch detection algorithm (PDA)
- Harmonic to noise ratio (HNR)

*Sound*
- Spectral centroid
- Spectral bandwidth
- Spectrograms (Linear, Log-frequency, Mel)

