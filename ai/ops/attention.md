
### **1. Input Setup**
Assume we have **3 input tokens** (e.g., words) with **embeddings of size 2**:
$
X = \begin{bmatrix}
1.0 & 0.5 \\
0.8 & 1.2 \\
0.3 & 0.9 \\
\end{bmatrix}
$

---

### **2. Learnable Matrices (Projection Matrices)**
Define **weights** for queries ($W_Q$), keys ($W_K$), and values ($W_V$), each of size $2 \times 2$ (for simplicity):
$
W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
W_K = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}, \quad
W_V = \begin{bmatrix} 1.5 & 0 \\ 0 & 1.5 \end{bmatrix}
$

---

### **3. Compute Q, K, V**
- **Queries**: $Q = X \cdot W_Q = X$ (identity weights preserve input here)  
- **Keys**: $K = X \cdot W_K = \begin{bmatrix} 0.5 & 0.25 \\ 0.4 & 0.6 \\ 0.15 & 0.45 \end{bmatrix}$  
- **Values**: $V = X \cdot W_V = \begin{bmatrix} 1.5 & 0.75 \\ 1.2 & 1.8 \\ 0.45 & 1.35 \end{bmatrix}$  

- **Queries and keys**:
They enable all possible combinations of words/tokens

- **Values**:
It contains features extracted from the sequence e.g. in one value matrix the featurs captured that are related to verbs and in the other value matrix the features extracted correspond to adjectives. 

- **Practical Example**:
For the input "The cat sat on the mat":
**Query** for "sat" might seek verbs and subjects.
**Key** for "cat" signals subject role.
**Value** for "cat" carries its subject-specific features.

---

### **4. Calculate Attention Scores**
For **each query vector** (row in $Q$), compute its dot product with **all keys** (rows in $K$):  
$
\text{Scores} = Q \cdot K^T = \begin{bmatrix}
1.0 \cdot 0.5 + 0.5 \cdot 0.25 & \cdots & 1.0 \cdot 0.15 + 0.5 \cdot 0.45 \\
\vdots & \ddots & \vdots \\
0.3 \cdot 0.5 + 0.9 \cdot 0.25 & \cdots & 0.3 \cdot 0.15 + 0.9 \cdot 0.45 \\
\end{bmatrix}
= \begin{bmatrix}
0.625 & 0.8 & 0.375 \\
0.8 & 1.04 & 0.48 \\
0.375 & 0.48 & 0.405 \\
\end{bmatrix}
$

- **Effectiveness**
The dot product of $Q \cdot K^T$ enables the consideration of all possible combinations of tokens and consequently the encoding the relevance of each combination (if a token is related to other token)

---

### **5. Scale and Softmax**
- **Scale** by $\sqrt{d_k}$ (here $d_k=2$):  
  $\text{Scaled Scores} = \frac{\text{Scores}}{\sqrt{2}} = \begin{bmatrix} 0.442 & 0.566 & 0.265 \\ 0.566 & 0.735 & 0.339 \\ 0.265 & 0.339 & 0.286 \end{bmatrix}$  
- **Softmax** (row-wise):  
  $\text{Attention Weights} = \text{Softmax}(\text{Scaled Scores}) \approx \begin{bmatrix} 0.35 & 0.42 & 0.23 \\ 0.35 & 0.45 & 0.20 \\ 0.32 & 0.38 & 0.30 \end{bmatrix}$ 
- **Interpretation** 
The first row of the attention weights shows the relevance of all other tokens (a word broken down to several tokens in this case two tokens) with the token of the first word. The second row shows the relevance of all other tokens with the token of the second word. This continues for all other words in the sequence. As we see, there are already redundancy in the attention weights because the relevance of the words is captured two times.   
---

### **6. Weighted Sum of Values**
Multiply attention weights with $V$ to get output $O$:  
$
O = \text{Weights} \cdot V = \begin{bmatrix}
0.35 \cdot 1.5 + 0.42 \cdot 1.2 + 0.23 \cdot 0.45 & \cdots \\
\vdots & \ddots \\
0.32 \cdot 1.5 + 0.38 \cdot 1.2 + 0.30 \cdot 0.45 & \cdots \\
\end{bmatrix}
= \begin{bmatrix}
1.16 & 1.74 \\
1.23 & 1.84 \\
1.05 & 1.58 \\
\end{bmatrix}
$

---

### **7. Interpretation**
- **Token 1's output** $[1.16, 1.74]$ combines information from all tokens, with **42% weight** provided by attention weights matrix (the attention weight on the fisrt row and second column) on the Tokens on the  second row of value matrix (most relevant).  
- **Scaling** prevents gradient vanishing/exploding.  
- **Softmax** ensures weights sum to 1, acting as a probabilistic mixture of values.

---

### **Key Equations**
1. **Attention**:  
   $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$  
2. **Multi-Head** (extension):  
   $\text{MultiHead} = \text{Concat}(\text{Attention}_1, \dots, \text{Attention}_h)W_O$  

---

### **Why This Works**
- **Dynamic Focus**: Tokens attend to contextually relevant parts (e.g., "bank" in "river bank" vs. "money bank").  
- **Parallelization**: Matrix operations are GPU-friendly.  
- **Interpretability**: Weights reveal token relationships (useful for explainability).  