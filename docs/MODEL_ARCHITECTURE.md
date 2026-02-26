# 🧠 Model Architecture: Spatio-Temporal Interaction Transformer

This document details the neural network architecture, mathematical formulations, and implementation strategies for the **Virtual Strategist**. The model is designed to process spatially-aligned telemetry stacks and tactical neighborhood data to predict race pace, risk, and tire degradation.

---

## 1. Overall Architecture Flow

The model is a multi-modal, multi-task network. It processes two distinct input streams and fuses them before branching into three specialized prediction heads.

1. **Input Stream A (Internal Health):** A 4D tensor of spatially-aligned historical telemetry `[Batch, 5_Laps, 100_Steps, 5_Features]`.
2. **Input Stream B (Tactical Environment):** A 3D tensor of relational gaps to the Lead and Chaser cars `[Batch, 2_Neighbors, 2_Features]`.

### **The Forward Pass**
1. **1D-CNN Stem:** Processes the `100_Steps` of each lap independently to extract high-level driving signatures, reducing the sequence length and projecting it into the model's hidden dimension ($d_{model}$).
2. **Spatio-Temporal Transformer:** Applies self-attention across the `5_Laps` dimension to calculate the exact lap-over-lap degradation delta.
3. **Tactical Fusion (Cross-Attention):** The output of the Transformer acts as the **Query**, while the Tactical Environment tensor acts as the **Keys and Values**. This allows the model to "explain away" pace drops caused by dirty air.
4. **Multi-Head Output:** The fused latent representation is passed through three separate Multi-Layer Perceptrons (MLPs).

---

## 2. Mathematical Formulation & Objective Functions

The model is optimized using a **Multi-Task Joint Loss Function**. By forcing the network to predict auxiliary tasks (like tire degradation and risk) alongside the main task (pace), we ensure the internal latent space is physically grounded.

### **The Joint Objective Function**
The total loss $\mathcal{L}_{total}$ is a weighted sum of three distinct loss components:

$$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{pace} + \lambda_2 \mathcal{L}_{risk} + \lambda_3 \mathcal{L}_{deg} $$

*(Typical weights: $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, $\lambda_3 = 2.0$ to heavily penalize ignoring tire wear).*

### **1. Pace Loss ($\mathcal{L}_{pace}$): Quantile Regression**
Predicting an "average" lap time is dangerous in F1. We need a probabilistic corridor. We use the **Asymmetric Laplace Loss (Pinball Loss)** to perform quantile regression for $\tau \in \{0.1, 0.5, 0.9\}$ (Optimistic, Median, Pessimistic pace).

$$ \mathcal{L}_{pace} = \frac{1}{N} \sum_{i=1}^N \sum_{\tau} \max(\tau(y_i - \hat{y}_i^{(\tau)}), (\tau - 1)(y_i - \hat{y}_i^{(\tau)})) $$

*Where $y_i$ is the true fuel-corrected pace, and $\hat{y}_i^{(\tau)}$ is the predicted pace at quantile $\tau$.*

### **2. Risk Loss ($\mathcal{L}_{risk}$): Cross-Entropy**
Predicts the probability of a discrete negative event on the next lap (e.g., being overtaken, or losing $>1.5s$ to traffic).

$$ \mathcal{L}_{risk} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{p}_{i,c}) $$

### **3. Degradation Loss ($\mathcal{L}_{deg}$): Mean Squared Error**
An auxiliary regression task predicting a continuous $0.0 - 1.0$ scalar representing latent tire health (derived from historical stint analysis).

$$ \mathcal{L}_{deg} = \frac{1}{N} \sum_{i=1}^N (h_i - \hat{h}_i)^2 $$

---

## 3. Hyperparameters

Below are the baseline hyperparameters for the Spatio-Temporal Transformer. These are optimized to balance expressive power with real-time inference latency.

| Component | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Data Engine Input** | `n_lap_context` | `5` | The number of historical laps in the input stack ($L, L-1, L-2, L-5, L_{base}$). |
| | `micro_sector_steps`| `100` | The spatial resolution of each lap (e.g., 1 sample per 3 meters over 300m). |
| **1D-CNN Stem** | `cnn_channels` | `[16, 32, 64]` | Progressively extracts higher-level features. |
| | `kernel_size` | `5` | Covers ~15 meters of track per convolution. |
| **Transformer** | `d_model` | `256` | The core hidden dimension size. |
| | `n_heads` | `8` | Number of attention heads (32 dim per head). |
| | `num_layers` | `4` | Number of stacked self-attention blocks. |
| | `dropout` | `0.1` | Prevents overfitting to specific driver quirks. |
| **Optimization** | `learning_rate` | `3e-4` | AdamW optimizer with Cosine Annealing. |
| | `batch_size` | `128` | Micro-sectors across multiple drivers/races. |

### **Architectural Constraint: Temporal Positional Embeddings**
It is critical to note that `n_lap_context` is a hard architectural constraint. Because the Transformer uses learnable positional embeddings to understand which input tensor is the "Current Lap" versus the "Baseline Lap," a model pre-trained with `n_lap_context = 5` cannot be fed a 10-lap stack during inference. The embedding layers are strictly dimensioned to the configuration set by the Data Engine.

---

## 4. Implementation Frameworks: PyTorch vs. Hugging Face

The architecture can be implemented using either pure PyTorch or leveraging the Hugging Face `transformers` library. The codebase supports a toggle between the two.

### **Option A: Pure PyTorch (Recommended for Production)**
* **Why:** The multi-modal nature of this model (CNN $\rightarrow$ Transformer $\rightarrow$ Cross-Attention) is highly custom. Pure `torch.nn` allows for maximum flexibility and easier compilation to TensorRT/ONNX for low-latency Triton serving.
* **Implementation:** Custom `nn.Module` classes for the `CNNStem`, `LapAttentionBlock`, and `TacticalFusionBlock`.

### **Option B: Hugging Face `transformers` (Recommended for Research)**
* **Why:** Allows us to wrap the model in the `PreTrainedModel` API, making it compatible with the Hugging Face ecosystem (Trainer API, Model Hub, easy checkpoint saving/loading).
* **Implementation:** We define a custom `VirtualStrategistConfig` and `VirtualStrategistModel` inheriting from `PreTrainedModel`. We utilize standard `BertEncoder` layers for the transformer blocks to save boilerplate code.

---

## 5. Training Strategy: Pre-Training vs. Fine-Tuning

Because F1 cars change drastically year-over-year (e.g., the 2022 ground-effect regulation changes), training from scratch on current-year data is difficult due to small sample sizes early in the season.

### **Phase 1: Foundation Pre-Training**
* **Data:** 5+ years of historical telemetry (e.g., 2018–2023).
* **Goal:** Teach the model the *universal physics of racing*—how tires degrade thermally, how aero-wash works, and how to interpret throttle/brake signatures.
* **Output:** A generalized `VirtualStrategist-Base` checkpoint.

### **Phase 2: Season/Car Fine-Tuning**
* **Data:** Current season data (or specific team data).
* **Goal:** Calibrate the model to the specific downforce levels, tire compound stiffness, and engine mapping of the current year.
* **Method:** We freeze the `1D-CNN Stem` (physics feature extraction) and fine-tune the `Transformer` and `Output Heads` using a lower learning rate (`1e-5`).