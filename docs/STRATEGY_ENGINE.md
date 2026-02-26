# ♟️ The Strategy Engine: Decision Making Under Uncertainty

This document details the **Strategy Engine**, the final layer of the Virtual Strategist. While the Spatio-Temporal Transformer predicts the *physics* of the car (pace and tire degradation), the Strategy Engine simulates the *tactics* of the race.

Its primary goal is to answer the most important question in Formula 1: **"When should we pit?"**

---

## 1. The Problem: The Curse of Determinism

If you ask a standard neural network, *"What is the mathematically fastest lap to pit?"*, it will give you a deterministic answer (e.g., "Lap 20"). 

**Why this fails in F1:**
Strategy is not about finding the fastest mathematical path in a vacuum; it is about finding the path with the highest probability of success under extreme uncertainty. 
* What if pitting on Lap 20 means exiting the pit lane exactly 0.5 seconds behind a slower car (the dreaded "DRS Train")?
* What if the driver locks up on Lap 18 and loses 2 seconds?

To solve this, the Strategy Engine abandons determinism and relies entirely on **Monte Carlo Rollouts** (stochastic simulation).

---

## 2. The Core Mechanic: Monte Carlo Rollouts

Instead of predicting one outcome, the engine uses the Transformer's outputs to run thousands of simulated futures for the remainder of the race.

### **The Simulation Loop (A Single Rollout)**
For every lap remaining in the race, the engine performs the following loop:
1. **State Update:** Fetch the current tire age, fuel load, and track position.
2. **Neural Forward Pass:** Query the Transformer for the predicted pace distribution. Because the model uses Quantile Regression, it outputs a probabilistic corridor (e.g., P10 Optimistic, P50 Median, P90 Pessimistic).
3. **Stochastic Sampling:** Roll a weighted die to pick the actual lap time from that corridor. (This introduces realistic variance, like a driver making a small mistake).
4. **Kinematic Update:** Move the car forward on the virtual track based on the sampled lap time.

---

## 3. Simulating the Pit Stop & Traffic (The Danger Zone)

The most dangerous moment in an F1 race is the pit exit. The Strategy Engine must perfectly simulate how a pit stop changes the car's physical relationship to the rest of the grid.

### **A Concrete Example: Avoiding the "Dirty Air" Trap**
Let’s assume it is **Lap 15**. 
* **Subject Car:** Lando Norris (P1)
* **Traffic:** Fernando Alonso (P5, **32 seconds** behind Lando).

We are running a simulation to test the hypothesis: *"What if Lando pits on Lap 20?"*

**1. Forward Simulation (Laps 15 $\rightarrow$ 19):**
Lando's tires degrade severely. By Lap 20, the gap to Alonso shrinks from **32 seconds** down to **26 seconds** (Alonso is catching up).

**2. The Pit Stop Event (Lap 20):**
Lando enters the pits in the simulation. The engine applies a static `Pit_Loss_Penalty` (e.g., +28.0 seconds at Silverstone). 
* Lando was **26 seconds** ahead.
* Lando loses 28 seconds in the pits.
* Lando exits the pits exactly **2.0 seconds behind** Alonso.

**3. The "Neighborhood" Recalculation:**
This is the critical step. Even though Alonso is P5 and Lando is P1, race position does not matter to aerodynamics. **Physical proximity matters.** 
The simulation recalculates the track order. It realizes Alonso is now physically in front of Lando. Alonso is instantly promoted to become Lando's **Lead Car**.

**4. The Aerodynamic Penalty (Input Stream B):**
*Why do we need to feed this gap back into the Neural Network?* The math knows Lando is 2.0s behind Alonso, but only the Neural Network knows *how much that hurts*.
The Strategy Engine feeds the new Relational Gap (`Delta_Distance = 2.0s`) into **Input Stream B** of the Transformer. The Transformer's Cross-Attention layer recognizes that Lando is now stuck in "Dirty Air" (Aero Wash) and severely penalizes his predicted pace for Lap 21, effectively ruining the advantage of his fresh tires.

---

## 4. Building the Strategic Value Surface

The engine does not just run this simulation once. It runs **10,000 rollouts** across every valid future pit window (e.g., testing Lap 16, Lap 17, Lap 18... up to Lap 30).

It aggregates these thousands of simulated futures into a **Strategic Value Surface**—a probability distribution table that allows the human strategist to weigh risk versus reward.

| Candidate Pit Lap | Expected Race Time (Median) | P90 Worst Case (Tail Risk) | Probability of Clean Air Exit |
| :--- | :--- | :--- | :--- |
| **Lap 18** | 1h 25m 10s | 1h 25m 45s *(High Risk of Traffic)* | 12% |
| **Lap 19** | 1h 25m 08s | 1h 25m 15s *(Clean Air Guaranteed)* | 85% |
| **Lap 20** | 1h 25m 15s | 1h 25m 20s *(Tire Cliff Reached)* | 90% |

**The Decision:** The engine recommends **Lap 19**. Even though pitting on Lap 18 *could* theoretically be fast, the Monte Carlo simulations reveal a massive "P90 Worst Case" tail risk (meaning in many simulations, the car exited the pits directly behind a slower car and ruined its race). Lap 19 provides the highest probability of a clean, fast race.

---

## 5. Real-Time Execution & Latency Budget

Running 10,000 simulations of a 50-lap race requires 500,000 neural network forward passes. Doing this in a standard Python `for` loop would take minutes—far too slow for a live F1 race where decisions must be made in milliseconds.

**Batched Rollouts:**
To meet the latency budget, the Strategy Engine leverages PyTorch's massive parallelization capabilities. Instead of simulating one rollout at a time, it constructs a massive batch tensor: `[10000_Rollouts, 5_Laps, 100_Steps, 5_Features]`. 

This allows the engine to compute the entire Monte Carlo tree for all candidate pit laps in a single GPU forward pass, returning the updated Value Surface to the pit wall in under 500 milliseconds.