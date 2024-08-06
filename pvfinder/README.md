
## Parallization Model 



### Current Model

```
Kernel Launch (Grid of Events)
│
├─── For each event (CUDA Block) in parallel:
│    │
│    ├─── Data Preprocessing (Parallel across tracks)
│    │    │ Each thread processes one or more tracks:
│    │    └─── Calculate ellipsoid parameters
│    │
│    ├─── For each interval (Sequential, but parallel across events):
│    │    │
│    │    ├─── Prepare interval input (Single-threaded)
│    │    │    └─── Collect and average track data for the interval
│    │    │
│    │    ├─── Neural network forward pass (Single-threaded)
│    │    │    └─── Process averaged input for the interval
│    │    │
│    │    └─── PV finding for the interval (Single-threaded)
│    │         └─── Identify potential PVs within the interval
│    │
│    ├─── Combine Interval Results (Single-threaded)
│    │    └─── Aggregate results from all intervals
│    │
│    └─── Final PV Finding (Single-threaded)
│         └─── Perform PV finding on combined histogram
│
└─── End Kernel
```

**Key Points:**

1. The main parallelism is at the event level, with each CUDA block processing one event independently.
2. Within each event, there's parallelism in the track preprocessing stage, where multiple threads work on different tracks simultaneously.
3. The interval processing, while sequential within an event, happens in parallel across different events.
4. The neural network forward pass and PV finding operations are not parallelized within an event, which could be an area for potential optimization.




### Goal Model

```
Kernel Launch (Grid of Events)
│
├─── For each event (CUDA Block) in parallel:
│    │
│    ├─── Data Preprocessing (Parallel across tracks)
│    │    │ Each thread processes one or more tracks:
│    │    └─── Calculate ellipsoid parameters
│    │
│    ├─── For each interval (Sequential, but parallel across events):
│    │    │
│    │    ├─── Neural Network Processing (Parallel across tracks)
│    │    │    │ Each thread for each relevant track:
│    │    │    ├─── Check if track is in interval
│    │    │    ├─── Normalize track input
│    │    │    ├─── Run neural network forward pass
│    │    │    └─── Accumulate output in shared memory
│    │    │
│    │    ├─── Synchronize threads
│    │    │
│    │    ├─── Combine Neural Network Results (Single-threaded)
│    │    │    └─── Aggregate outputs from all threads
│    │    │
│    │    └─── PV finding for the interval (Single-threaded)
│    │         └─── Identify potential PVs within the interval
│    │
│    ├─── Synchronize threads
│    │
│    ├─── Combine Interval Results (Single-threaded)
│    │    └─── Aggregate results from all intervals
│    │
│    └─── Final PV Finding (Single-threaded)
│         └─── Perform PV finding on combined histogram
│
└─── End Kernel
```

**Key points of parallelization:**

1. Event-level parallelism:
   - Each CUDA block processes one event independently.
   - This allows multiple events to be processed simultaneously across different blocks.

2. Track-level parallelism within each event:
   - During data preprocessing: Threads work in parallel to calculate ellipsoid parameters for tracks.
   - During neural network processing: Threads process individual tracks in parallel for each interval.

3. Interval processing:
   - Intervals are processed sequentially within each event.
   - However, different events process their intervals in parallel.

4. Neural network forward pass:
   - Now parallelized across tracks within each interval.
   - Each thread processes one or more tracks, running the full neural network for each.

5. Result accumulation:
   - Parallel reduction: Threads accumulate their results in shared memory.
   - Final combination is done by a single thread per interval.

6. PV finding and result combination:
   - These steps remain single-threaded within each event but occur in parallel across events.

