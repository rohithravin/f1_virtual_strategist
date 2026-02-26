# 🏎️ The Virtual Strategist: Neighborhood Interaction Model

An end-to-end inference system designed to predict Formula 1 race pace and optimal pit windows. This project implements a **Multi-Modal Spatio-Temporal Transformer** that fuses spatially-aligned subject-car telemetry with the tactical environment of the immediate "Neighborhood" to optimize race completion time.

---

## 🛠️ System Architecture

### 1. [The Interaction Data Engine](docs/DATA_ENGINE.md)
* **Spatial Micro-Sectors:** Telemetry is sliced by **Distance** (not time) to ensure perfect physical alignment of track features across multiple laps.
* **Historical Stacking:** Feeds the model an exponential look-back stack ($L, L-1, L-2, L-5, L_{baseline}$) to calculate the exact derivative of tire degradation.
* **Neighborhood Synchronization:** A custom `SynchroEngine` that aligns asynchronous GPS and timing data for the subject car, the **Lead** car (aero-interference), and the **Chaser** car (defensive pressure).
* **Fuel-Corrected Delta:** A normalization pipeline that decouples tire degradation from fuel burn-off weight loss to find "True Pace."

### 2. [Model Architecture & Layout](docs/MODEL_ARCHITECTURE.md)

```text
INPUT STREAMS                          MODEL BACKBONE                          OUTPUT HEADS
================                      ================                        ================

[ Spatial Micro-Sector Stack ]        +------------------+
(5 Laps x 100 distance steps)         |  1D-CNN Encoder  |
       |                              | (Feature Extr.)  |
       +----------------------------> +------------------+
                                               |
                                               v
                                      +------------------+
                                      | Spatio-Temporal  |
                                      |   Transformer    |
                                      | (Self-Attention) |
                                      +------------------+
                                               |
                                               v                    +------> [ Pace Head ]
[ Relational Gaps  ]                  +------------------+          |        (P10, P50, P90)
(Lead & Chase Gaps) ----------------> |   Fusion Layer   | ---------+
                                      | (Cross-Attention)|          |------> [ Risk Head ]
                                      +------------------+          |        (Prob. Overtake)
                                                                    |
                                                                    +------> [ Degradation ]
                                                                             (Latent Health)
```

### 3. [The Strategy Engine](docs/STRATEGY_ENGINE.md)
* **Monte Carlo Rollouts:** Simulates 10,000 future race states to calculate the highest probability of success under uncertainty.
* **Traffic Evaluation:** Dynamically recalculates the "Neighborhood" during simulated pit stops to avoid exiting into "Dirty Air" (DRS Trains).
* **Strategic Value Surface:** Outputs a probabilistic decision matrix weighing Expected Race Time against P90 Tail Risk.

## 🏗️ Detailed Architecture Breakdown

### **Phase 1: Feature Extraction (The Encoder)**
The **1D-CNN Stem** acts as a learnable signal processor. It processes the spatially-resampled telemetry (e.g., 1 sample every 3 meters) and applies convolutions to identify "driving signatures."
* **Kernel Size (3-5):** Captures the relationship between immediate sensor changes (e.g., the transition from 100% brake to 10% trail-braking over a 15-meter stretch).
* **Max-Pooling:** Reduces noise while preserving the most extreme stress events (e.g., G-force peaks or sudden spikes in RPM).
* **Why this route?** Telemetry is notoriously "noisy". A CNN serves as a superior trainable filter compared to a standard linear projection, ensuring the Transformer receives clean physical features.

### **Phase 2: Spatio-Temporal Modeling (The Transformer)**
The heart of the model is a **Transformer Encoder block**. Because the input is a stack of spatially-aligned laps, the self-attention mechanism gains a superpower: track geometry is "explained away."
* **Lap-Over-Lap Attention:** The model compares Data Point #80 (e.g., the exact apex of Turn 4) across Lap 5, Lap 20, and Lap 25 simultaneously. 
* **Degradation Signatures:** It learns that if throttle application at distance $X$ was 100% on Lap 5, but only 65% with spiking RPMs on Lap 25, the tires have lost thermal grip.
* **Why this route?** If we only looked at a single window in time, the model wouldn't know if the car was slow due to dead tires or simply navigating a hairpin. Stacking history by distance isolates the true degradation delta.

### **Phase 3: Tactical Fusion (Cross-Attention)**
This layer is where the car interacts with its neighbors. It allows the model to differentiate between internal mechanical issues and external tactical constraints.
* **Query (Q):** Represents the subject car's internal status (e.g., "I am sliding in Sector 2").
* **Keys (K) / Values (V):** Represent the Lead and Chaser positions and velocities.
* **Mechanism:** If the Query (Subject) shows a struggle for grip and the Keys (Lead) show a < 0.5s gap, the Fusion Layer assigns high weight to **"Aero Wash"** as the cause of pace loss rather than natural tire wear.

### **Phase 4: Multi-Head Prediction**
The latent representation is fed into three separate Linear heads to provide actionable insights:
1.  **Pace Head:** Uses a 3-unit output for **Quantile Regression** ($P_{10}, P_{50}, P_{90}$) to provide a confidence interval for the next lap pace.
2.  **Risk Head:** A **Softmax layer** predicting the probability of a discrete "Event" (e.g., being overtaken or a mechanical failure).
3.  **Degradation Head:** A Regressor predicting a $0-1$ scalar of **latent tire health**, utilized to trigger optimal pit entry recommendations.

---

## 💡 Key Engineering Decisions

* **Distance Over Time:** We slice telemetry by **Track Distance (Meters)** rather than Time (Seconds). A 10-second window covers different amounts of track depending on car speed, leading to misaligned tensors. Distance-based resampling guarantees apples-to-apples comparisons of specific corners.
* **Exponential Historical Stacking:** Instead of feeding 50 previous laps (which destroys memory) or just 1 previous lap (which misses macro-trends), we feed an exponential stack: $L, L-1, L-2, L-5, L_{baseline}$. This provides immediate derivatives while maintaining a high-grip anchor point.
* **Neighborhood Constraints:** By limiting spatial attention to the immediate Lead and Chaser, we eliminate the computational overhead of a 20-car global rollout while capturing 90% of the tactical variance.