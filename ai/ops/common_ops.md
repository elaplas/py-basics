# Part 1

---

## 🔧 **1. Convolution (Conv2D / Conv1D / Conv3D)**

### 🧮 Math:
\[
Y(i,j) = \sum_m \sum_n X(i+m, j+n) \cdot K(m,n)
\]

- \( X \): input (e.g., image), \( K \): kernel  
- Sliding dot product

### 🧠 Why:
- Detects **spatial features** like edges, textures  
- Core to CNNs in vision tasks

---

## 📐 **2. Matrix Multiplication (MatMul / Dense Layers)**

### 🧮 Math:
\[
y = Wx + b
\]

- \( W \): weight matrix  
- \( x \): input vector  
- \( b \): bias

### 🧠 Why:
- Core to all neural layers  
- Computes **linear transformations**

---

## 🔁 **3. ReLU (Rectified Linear Unit)**

### 🧮 Math:
\[
\text{ReLU}(x) = \max(0, x)
\]

### 🧠 Why:
- Adds **non-linearity**  
- Sparse gradients, avoids vanishing gradients

---

## 📊 **4. Softmax**

### 🧮 Math:
\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

### 🧠 Why:
- Turns logits into **probabilities**  
- Used in classification

---

## 🧲 **5. Sigmoid**

### 🧮 Math:
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

### 🧠 Why:
- Squashes input to (0, 1)  
- Good for **binary classification**, gates in RNNs

---

## 🌀 **6. Batch Normalization**

### 🧮 Math:
\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}};\quad y = \gamma \hat{x} + \beta
\]

- Normalize, then scale and shift

### 🧠 Why:
- Stabilizes training  
- Enables higher learning rates

---

## ⛓ **7. Max Pooling / Avg Pooling**

### 🧮 Math:
\[
y = \max_{i,j \in \text{window}} X(i,j) \quad \text{or} \quad \frac{1}{N} \sum X(i,j)
\]

### 🧠 Why:
- Downsamples input  
- Retains most important features

---

## 📏 **8. Dropout**

### 🧮 Math:
\[
y_i = x_i \cdot r_i, \quad r_i \sim \text{Bernoulli}(p)
\]

- Randomly sets neurons to 0 during training

### 🧠 Why:
- Prevents overfitting  
- Acts like ensembling

---

## 🧠 **9. Layer Normalization**

### 🧮 Math:
\[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}};\quad y_i = \gamma \hat{x}_i + \beta
\]

- Normalizes **per sample**, not per batch

### 🧠 Why:
- More stable in NLP (e.g., transformers)

---

## ⏪ **10. Residual Connection**

### 🧮 Math:
\[
y = F(x) + x
\]

### 🧠 Why:
- Solves **vanishing gradient** problem  
- Enables deep networks (ResNet, transformers)

---

## 🔄 **11. Attention**

### 🧮 Math (Scaled Dot-Product Attention):
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

### 🧠 Why:
- Lets the model **focus** on relevant parts of the input  
- Core of transformers

---

## 🧮 **12. Embedding Lookup**

### 🧮 Math:
\[
\text{Embedding}(i) = W[i]
\]

- Like selecting a row from a matrix

### 🧠 Why:
- Turns discrete tokens (e.g., words) into **dense vectors**

---

## 🧵 **13. Concatenation**

### 🧮 Math:
\[
y = [x_1, x_2]
\]

- Joins along a given dimension

### 🧠 Why:
- Combine features or model outputs (e.g., skip connections)

---

## 📐 **14. Layer Scaling / Parameter Scaling**

### 🧮 Math:
\[
y = \alpha \cdot x, \quad \alpha \in \mathbb{R}
\]

- Optional scalar or vector parameter

### 🧠 Why:
- Helps in stabilizing **deep transformers**  
- Added in modern designs like GPT-3, T5

---

## 🧊 **15. GELU (Gaussian Error Linear Unit)**

### 🧮 Math (approximation):
\[
\text{GELU}(x) = 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right]\right)
\]

### 🧠 Why:
- Smoother than ReLU  
- Standard in transformer models (BERT, GPT)

---

## ✅ Summary Table

| Operation          | Use Case / Layer Type       | Why It's Important                           |
|-------------------|-----------------------------|----------------------------------------------|
| Conv2D            | CNNs (Vision)               | Feature extraction from images               |
| MatMul            | Fully connected layers      | Linear transformations                       |
| ReLU              | All layers                  | Non-linearity, sparsity                      |
| Softmax           | Output layer (classifier)   | Probability distribution                     |
| Sigmoid           | Binary classification, gates| Activation, binary output                    |
| BatchNorm         | CNNs, MLPs                  | Stable training                              |
| Pooling           | CNNs                        | Downsampling                                 |
| Dropout           | All                         | Regularization                               |
| LayerNorm         | NLP (transformers)          | Normalize per token/sample                   |
| Residual          | Deep nets, transformers     | Stable deep training                         |
| Attention         | NLP, Vision                 | Focus mechanism                              |
| Embedding         | NLP                         | Token representation                         |
| Concatenation     | Multi-modal, skip links     | Combine info                                 |
| Scaling           | Transformers                | Stabilize gradients                          |
| GELU              | Transformers (default)      | Smooth activation, better gradient flow
---

# Part 2

---

## 🧮 **1. Tanh (Hyperbolic Tangent)**

### Math:
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

### Why:
- Activation that squashes inputs to \([-1, 1]\)  
- Used in RNNs (e.g., LSTM, GRU) for smooth gradients  
- Zero-centered unlike sigmoid

---

## ⌛ **2. LSTM Cell (Long Short-Term Memory)**

### Math:
\[
\begin{align*}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
c_t &= f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t \\
h_t &= o_t \cdot \tanh(c_t)
\end{align*}
\]

### Why:
- Captures **long-term dependencies** in sequences  
- Reduces vanishing gradient problems in RNNs

---

## 🔄 **3. GRU (Gated Recurrent Unit)**

### Math:
\[
\begin{align*}
z_t &= \sigma(W_z x_t + U_z h_{t-1}) \\
r_t &= \sigma(W_r x_t + U_r h_{t-1}) \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \cdot h_{t-1})) \\
h_t &= (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t
\end{align*}
\]

### Why:
- Simpler than LSTM  
- Good for sequence modeling with fewer parameters

---

## 🪞 **4. Transpose / Permute / Reshape**

### Math:
- Change shape without changing data:
  \[
  X \in \mathbb{R}^{B \times C \times H \times W} \rightarrow \mathbb{R}^{B \times H \times W \times C}
  \]

### Why:
- Needed for aligning tensors in attention, image processing, or framework-specific formats (e.g., PyTorch vs TensorFlow)

---

## 📦 **5. Flatten**

### Math:
\[
\text{Flatten}(X) \in \mathbb{R}^{C \times H \times W} \rightarrow \mathbb{R}^{C \cdot H \cdot W}
\]

### Why:
- Connects CNN features to fully connected layers  
- Simple reshape operation

---

## 🧬 **6. Upsampling / Interpolation**

### Math:
\[
\text{Upsample}(X) = X' \quad \text{where } \text{size}(X') > \text{size}(X)
\]

### Common methods: nearest, bilinear, bicubic interpolation

### Why:
- Used in image generation (e.g., GANs, UNet)  
- Reverse of pooling, restores spatial resolution

---

## 🔀 **7. Argmax / Argmin**

### Math:
\[
\text{argmax}(x) = \underset{i}{\arg\max} \, x_i
\]

### Why:
- Selects the **most likely class**  
- Used in classification, attention selection, reinforcement learning

---

## 🧱 **8. Padding (Zero / Reflect / Replicate)**

### Math:
If \( X \in \mathbb{R}^{H \times W} \), zero-padding adds zeros around the borders.

### Why:
- Preserves input size in conv layers  
- Prevents shrinking during convolution  
- Reflect/replicate helps avoid artifacts in images

---

## 🎛 **9. GroupNorm / InstanceNorm**

### Math (InstanceNorm example):
\[
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_{nc}}{\sqrt{\sigma^2_{nc} + \epsilon}}
\]

### Why:
- Alternatives to BatchNorm for small batch sizes or style transfer  
- GroupNorm groups channels for more stable training

---

## 📦 **10. Clipping (Gradient Clipping / Value Clipping)**

### Math:
\[
x = \text{clip}(x, a, b) \Rightarrow \max(a, \min(x, b))
\]

Or for gradients:
\[
\text{if } ||g|| > \tau: \quad g = \frac{\tau}{||g||} \cdot g
\]

### Why:
- Prevents **exploding gradients** in RNNs  
- Keeps values numerically stable

---

## ✅ Summary Table

| Operation         | Use Case                           | Why It Matters                             |
|------------------|-------------------------------------|--------------------------------------------|
| Tanh             | RNN activations                     | Smooth and zero-centered                   |
| LSTM             | Sequence modeling                   | Captures long-term dependencies            |
| GRU              | Compact RNN                         | Simpler than LSTM, similar power           |
| Transpose        | Tensor alignment                    | Matches dimensions in models               |
| Flatten          | CNN to FC layers                    | Reshape for classification                 |
| Upsampling       | GANs, segmentation                  | Increase resolution                        |
| Argmax           | Classification, attention heads     | Choose most likely option                  |
| Padding          | CNNs, audio                         | Preserve size, avoid border loss           |
| Group/InstanceNorm | Style transfer, small batch training | More stable normalization                 |
| Clipping         | RNNs, gradients                     | Keeps training stable                      |
