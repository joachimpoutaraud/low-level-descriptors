# Audio low-level descriptors (LLDs)

This module encapsulates multiple audio low-level descriptors into a streamlined and modular implementation.

## Prerequisites

- librosa
- matplotlib
- numpy

## Importing low-level descriptors

```python
import sys
!{sys.executable} -m pip install -r requirements.txt -q

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

*Sound*
- Spectral centroid
- Spectral bandwidth
- Spectrograms (Linear, Log-frequency, Mel)

