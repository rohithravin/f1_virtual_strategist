# ⚙️ The Interaction Data Engine: Deep Dive

This document outlines the technical architecture of **The Interaction Data Engine**, the foundational preprocessing layer of the Virtual Strategist. 

The goal of this engine is simple but technically complex: **Transform asynchronous, multi-rate, noisy Formula 1 telemetry into spatially-aligned, physics-corrected tensors for real-time Transformer inference.**

Formula 1 data is notoriously difficult to align. A subject car's internal telemetry (throttle, brake) may sample at 10Hz, while GPS positional data for rival cars might update at 3Hz, and timing loops trigger only three times a lap. The Interaction Data Engine bridges these gaps to construct the "Neighborhood" and stack the "History."

---

## 1. The Spatial Telemetry Pipeline (Distance-Based Slicing)

The core input to the `1D-CNN Encoder` is the mechanical stress profile of the subject car. However, if we slice telemetry by **Time** (e.g., a 10-second window), the model will compare 500 meters of track on a fast lap with only 400 meters of track on a slow lap. It will compare a straightaway to a corner, leading to catastrophic failure.

To fix this, the Data Engine slices telemetry by **Track Distance (Meters)**.

### **Data Ingestion & Resampling**
1. **Raw Ingestion:** Fast-stream ingestion of `[RPM, Speed, Throttle, Brake, Gear]` alongside the cumulative `Distance` channel.
2. **Standardization:**
   * **Brake & Throttle:** Normalized to a continuous $[0.0, 1.0]$ scale.
   * **RPM:** Scaled relative to the engine's hard-limiter to provide a standardized power-band metric.
3. **Spatial Interpolation:** We utilize **Cubic Spline Interpolation** over the `Distance` channel to guarantee exactly 1 sample every fixed distance interval (e.g., every 3 meters). 
4. **Output:** A dense vector $\Phi_{d} \in \mathbb{R}^{5}$ for a specific "Micro-Sector" (e.g., 300 meters long, resulting in exactly 100 data points).

---

## 2. Historical Stacking (The Look-Back Strategy)

To calculate the *derivative* of pace (the true rate of degradation), the model needs to see how the car performed in the exact same Micro-Sector on previous laps. 

If we feed it every single lap, memory explodes. If we only feed it the previous lap, it misses the macro-trend. The Data Engine solves this using an **Exponential Decay Look-Back Stack**.

### **Configuration: `lookback_indices`**
The Data Engine is configured with a specific number of historical laps to extract, defined by `lookback_indices = [0, 1, 2, 5, "baseline"]`. This results in an `n_lap_stack = 5`.

For any given Micro-Sector on the current Lap $L$, the engine fetches a 5-Lap stack:
1.  **$L$ (Current Lap, Index 0):** The raw, real-time state.
2.  **$L-1$ (Previous Lap, Index 1):** To calculate the immediate derivative (e.g., "Did I just lock up and flat-spot the tire?").
3.  **$L-2$ (Two Laps Ago, Index 2):** Confirms if $L-1$ was a fluke (traffic) or a real trend.
4.  **$L-5$ (Medium History, Index 5):** Shows the macro-trend of the stint.
5.  **$L_{baseline}$ (The "Out Lap" + 2):** The fastest clean lap on this set of tires when they were fresh. This serves as the absolute physical baseline for what the car is capable of.

### **Handling Early-Stint Edge Cases**
What happens on Lap 3 when $L-5$ does not exist? The Data Engine handles this via **Baseline Padding**. If a historical index is out of bounds for the current tire stint, the engine duplicates the $L_{baseline}$ tensor into that slot and passes a binary `padding_mask` to the model so the Transformer knows to ignore the artificial delta.

Because Data Point #80 is the exact same inch of tarmac across all 5 laps, the Transformer's Self-Attention mechanism can instantly identify if the driver is applying less throttle or experiencing more wheel-spin (RPM spikes) compared to the baseline.

---

## 3. The `SynchroEngine` (Neighborhood Alignment)

The most complex component of the pipeline is identifying and tracking the **Neighborhood**—the car immediately ahead (Lead) and immediately behind (Chaser).

### **Dynamic Role Assignment**
The engine cannot assume the car ahead at Lap 10 is the same car ahead at Lap 11. The `SynchroEngine` maintains a lightweight state machine that dynamically recalculates the Lead and Chaser at every timing loop (Sector 1, Sector 2, Start/Finish) or via GPS cross-referencing.
* **Edge Cases Handled:** Pit stops, blue flags (lapped cars), and side-by-side racing. If a Lead car pits, the engine instantly promotes the next car on track to the Lead position in the tensor.

### **Temporal Alignment (Solving Asynchronous Rates)**
If the subject car is sampled exactly at $t = 12.0s$, the GPS data for the Lead car might only exist at $t = 11.9s$ and $t = 12.2s$.
* **The Solution:** The `SynchroEngine` performs **Linear Kinematic Extrapolation**. It uses the Lead car's last known velocity vector and heading to project its exact $X, Y$ coordinate at precisely $t = 12.0s$.

### **Calculating Relational Gaps**
Absolute coordinates ($X, Y$ on a map) are useless to a neural network because they change based on the track. The engine converts absolute GPS into **Relative Delta Tensors**:
$$ r_{lead} = [ \Delta \text{Distance}, \Delta \text{Velocity} ] $$
$$ r_{chaser} = [ \Delta \text{Distance}, \Delta \text{Velocity} ] $$

*   **$\Delta$ Distance:** Provides the severity of "Aero Wash" (dirty air). If $\Delta \text{Dist} < 1.0s$, the model knows the subject car is losing downforce.
*   **$\Delta$ Velocity (Closing Speed):** Identifies if the subject car is attacking (positive closing speed) or defending (negative closing speed).

---

## 4. Fuel-Corrected Delta (Physics Normalization)

Formula 1 cars start a race with ~110kg of fuel and end near 0kg. As the car gets lighter, it naturally gets faster (roughly $0.03s$ to $0.06s$ per lap, per kg of fuel burned).

If the model is fed raw lap times, it will fundamentally misunderstand tire wear. It will see lap times improving and assume the tires are getting *better*, completely ignoring that the weight loss is masking the severe thermal degradation of the rubber.

### **The "True Pace" Algorithm**
The engine applies a physical correction to calculate "True Pace":
1. **Estimate Mass:** Calculate current car mass $M_t = M_{start} - (\text{Lap} \times \text{Fuel Burn Rate})$.
2. **Calculate Weight Penalty:** $W_{penalty} = f(M_t, \text{Track Specific Weight Multiplier})$.
3. **Normalize:** `True_Pace = Raw_Lap_Time + W_penalty`

By passing `True_Pace` to the model, the Transformer's `Degradation Head` can accurately isolate the performance loss caused strictly by the tires.

---

## 5. Storage Architecture (Parquet & DuckDB)

To train a Transformer with high-frequency data across thousands of historical races, row-based databases (like PostgreSQL) are too slow.

### **Columnar Superiority**
We utilize **DuckDB** querying over partitioned **Parquet** files. DuckDB's vectorized execution engine is uniquely suited for aggregation.
* **Partitioning:** Data is heavily partitioned by `Year / Race / Session / Driver`.

### **Zero-Copy Windowing**
During training and inference, the model requires overlapping spatial windows. DuckDB allows us to perform high-speed `OVER (ORDER BY distance ROWS BETWEEN 99 PRECEDING AND CURRENT ROW)` queries. Because DuckDB interfaces natively with Python (and Arrow), these sliding windows are passed directly into PyTorch/MLX tensors with **zero-copy memory overhead**.

---

## 6. Final Output: The Model Tensor Schema

At the end of the pipeline, the Data Engine produces a synchronized batch of tensors ready for the Transformer. For a given Micro-Sector $d$ and a batch size $B$:

```python
# 1. Subject Telemetry Stack (Input to 1D-CNN)
# Shape: [Batch, History_Laps, Sequence_Length, Features]
X_telemetry = Tensor(B, 5, 100, 5)  
# History_Laps: [L, L-1, L-2, L-5, L_baseline]
# Sequence_Length: 100 distance steps (e.g., a 300m Micro-Sector)
# Features: [Throttle, Brake, RPM, Speed, Gear]

# 2. Tactical Neighborhood State (Input to Cross-Attention)
# Shape: [Batch, Num_Neighbors, Relational_Features]
X_tactical = Tensor(B, 2, 2)     
# Neighbors: [Lead, Chaser]
# Relational_Features: [Delta_Distance_Sec, Delta_Velocity_Kmh]

# 3. Normalized Targets (For Loss Calculation)
Y_pace = Tensor(B, 1) # Fuel-Corrected "True Pace" next lap
```

The Data Engine ensures that every element in `X_tactical` is perfectly time-aligned with the exact final tick of the `X_telemetry` window, allowing the Cross-Attention layer to accurately fuse internal mechanics with external strategy.