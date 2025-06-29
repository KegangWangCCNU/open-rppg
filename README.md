# open-rppg
An easy-to-use rPPG inference toolbox

## Installation 
Python >= 3.9
```bash
pip install open-rppg
```
## Import and Use 
```python
import rppg
model  = rppg.Model()
result = model.process_video("your_video.mp4")
```

## Result Example 
```python
{'hr': 100.45004500450045,        # Heart Rate (FFT Method)
 'SQI': 0.8665749931341046,       # Signal Quality
 'hrv':{
    'bpm': 103.6194862200475,     # Heart Rate (Peak Method)
    'ibi': 579.0416666666666,     # Inter-Beat Interval
    'sdnn': 54.76628055757589,    # Standard Deviation of NN intervals
    'sdsd': 30.674133962201175,   # Standard Deviation of Successive Differences
    'rmssd': 46.25344260031846,   # Root Mean Square of Successive Differences
    'pnn20': 0.5714285714285714,  # Proportion of NN50 > 20ms
    'pnn50': 0.2857142857142857,  # Proportion of NN50 > 50ms
    'hr_mad': 8.333333333333314,  # Heart Rate Median Absolute Deviation
    'sd1': 29.276576197229755,    # Short-term variability
    'sd2': 59.75143144804733,     # Long-term variability
    's': 5495.642490576809,       # Poincaré Plot Area
    'sd1/sd2': 0.489972800445545, # SD1/SD2 Ratio
    'breathingrate': 0.21607605877268798,
    'VLF': 0.09521664913596516,   # Very Low Frequency Power
    'TP': 2.056694418632364,      # Total Power
    'HF': 1.2267116642737315,     # High Frequency Power
    'LF': 0.7347661052226675,     # Low Frequency Power
    'LF/HF': 0.5989721355243509   # LF/HF Ratio
  },
 'latency': 0.0}                  # Real-Time Latency
```

## Real-Time Mode 
```python
import time
model = rppg.Model()

with model.video_capture(0):          # Connect to your webcam
    while True:
        result = model.hr(start=-15)  # Get heart rate from last 15 seconds
        if result:
            print(f"Heart Rate: {result['hr']} BPM")
        time.sleep(1)
```

## Real-Time Frame Preview

```python
for frame, box in model.preview:     # Current RGB frame and detection box
    x, y  = box                      
    face  = frame[x[0]:x[1], y[0]:y[1]]
```

## Get BVP Wave 
```python
bvp, ts        = model.bvp()         # BVP with timestampes
raw_bvp, ts    = model.bvp(raw=True) # Unfiltered BVP
```

## Time Slice 
```python
now       = model.now                      # Video duration or current time
bvp, ts   = model.bvp(start=10, end=20)    # BVP slice from 10 to 20 seconds
bvp, ts   = model.bvp(start=-15)           # The last 15-second slice
hr        = model.hr(start=-15)            # HR of the last 15 seconds 
```

## Model Selection 
```python
print(rppg.supported_models) # ['ME-chunk.rlap', 'ME-flow.rlap', .......]
model = rppg.Model('RhythmMamba.rlap') # RhythmMamba trained on rlap
```
## Pretrained Models 
| Model | Training Set | Description | Paper |
|-|-|-|-| 
|ME-chunk|PURE RLAP|rPPG based on state-space model|2025|
|ME-flow|PURE RLAP|ME in low-latency real-time mode|2025| 
|PhysMamba|PURE RLAP|Mamba with fast-slow network|2024|
|RhythmMamba|PURE RLAP|Mamba with 1D FFT|2025|
|PhysFormer|PURE RLAP|Transformer with central diff conv|2022| 
|TSCAN|PURE RLAP|Conv attention with temporal shift|2020|
|EfficientPhys|PURE RLAP|TSCAN with self attention|2022|
|PhysNet|PURE RLAP|3D CNN encoder-decoder network|2019| 

## Use CUDA 
Install JAX with CUDA (Linux only).
```bash
pip install jax[cuda]
```
