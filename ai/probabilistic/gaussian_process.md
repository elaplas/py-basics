
##### **1. Infinite-Dimensional Nature**
A Gaussian Process (GP) is a distribution over **functions**. For any input domain (e.g., real numbers $x \in \mathbb{R}$), the GP defines a joint Gaussian distribution (multi-variate normmal distribution) over **all possible function values** $f(x)$ across infinitely many inputs $x$. In fact, each realization of function $f_i$ for the input $x_i$ adds a new dimesion to the joint Gaussian distribution. This makes the joint distribution "infinite-dimensional" because:
- The input space (e.g., $x \in \mathbb{R}$) is infinite. Consequantly the realization of function $f: \mathcal{X} \to \mathbb{R}$, becomes infinite.

##### **2. Finite-Dimensional Projections**
In practice, we work with **finite subsets** of the input domain. For any $n$ points $\{x_1, ..., x_n\}$, the corresponding function values $[f(x_1), ..., f(x_n)]$ follow a multivariate normal distribution:
$$
\begin{bmatrix}
f(x_1) \\
\vdots \\
f(x_n)
\end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix}
m(x_1) \\
\vdots \\
m(x_n)
\end{bmatrix},
\begin{bmatrix}
k(x_1, x_1) & \cdots & k(x_1, x_n) \\
\vdots & \ddots & \vdots \\
k(x_n, x_1) & \cdots & k(x_n, x_n)
\end{bmatrix}
\right)
$$
where $m(x_i)$ is the mean of function value $f(x_i)$ and $k(x_i, x_j)$ is the kernel to simulate the variance/covariance between function value pairs $f(x_i)$ and $f(x_j)$. Notice that in GP the function values $f(x_i)$ are considered as random variables and the input values $x_i$ as deterministic and non-random variables.

##### **3. Kernel Function as the Key**
The kernel $k(x_i, x_j)$ encodes **prior assumptions** about the function's behavior (e.g., smoothness, periodicity). For example:
- **Squared Exponential (RBF) kernel**:  
  $k(x_i, x_j) = \sigma^2 \exp\left(-\frac{(x_i - x_j)^2}{2l^2}\right)$  
  Produces infinitely differentiable, smooth functions.
- **Matérn kernel**:  
  $k(x_i, x_j) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}|x_i - x_j|}{l}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}|x_i - x_j|}{l}\right)$ 
  Generates less smooth functions (controlled by $\nu$).

##### **4. Conditioning on Data**
When we observe data $\{x_i, y_i\}_{i=1}^n$, the GP **updates its predictions** using Bayes' rule. The posterior distribution at a new point $x_*$ is Gaussian with:  
$$
\begin{aligned}
\text{Mean} &= m(x_*) + \mathbf{k}_*^\top (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{y} \\
\text{Variance} &= k(x_*, x_*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{k}_*
\end{aligned}
$$  
where:
- $\mathbf{K}$: Kernel matrix for all input points.
- $\mathbf{k}_*$: Kernel vector between $x_*$ and all input points.
- $\sigma^2$: Observation noise.
- $ k(x_*, x_*)$ the kernel at input point $x_*$
- $m(x_*)$ the mean of function value $f(x_*)$ and it is usually considered to be zero for simplicity.
- Mean and Variance are the predicted function value and the corresponding uncertainity at input value $x_*$


##### **5. Practical Handling of "Infinite-Dimensional" Data**
- **Kernel trick**: The kernel implicitly maps inputs to an infinite-dimensional feature space, but computations remain in the finite-dimensional space of observed data.
- **Sparse approximations**: Techniques like inducing points reduce computational complexity by approximating the full GP with a subset of pseudo-inputs.
- **Function space priors**: The GP prior ensures that any finite-dimensional projection is Gaussian, enabling tractable inference.

### Example
Here's how to manually calculate a Gaussian process (GP) regression for classifier accuracy over batch size and learning rate, assuming **5 data points** with batch sizes, learning rates, and accuracies. We'll use a simplified RBF kernel and focus on **predicting accuracy for a new (batch size, learning rate)** pair.

---

##### **1. Define the Data**
Assume the following measurements (batch size, learning rate, accuracy):  
$$  
\begin{align*}  
\mathbf{X} &= \begin{bmatrix} 32 & 0.001 \\ 64 & 0.01 \\ 128 & 0.005 \\ 256 & 0.02 \\ 512 & 0.001 \end{bmatrix}, \quad  
\mathbf{y} = \begin{bmatrix} 0.82 \\ 0.85 \\ 0.88 \\ 0.83 \\ 0.80 \end{bmatrix}  
\end{align*}  
$$  

##### **2. Choose a Kernel**  
Use the **squared exponential (RBF)** kernel with length scales $l_1$ (batch size) and $l_2$ (learning rate):  
$$  
k(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{(x_{i1} - x_{j1})^2}{2l_1^2} - \frac{(x_{i2} - x_{j2})^2}{2l_2^2}\right)  
$$  
**Assume**: $l_1 = 100$ (batch size scale), $l_2 = 0.1$ (learning rate scale), and $\sigma_n^2 = 0.01$ (noise variance).

##### **3. Compute the Covariance Matrix**  
Calculate the **5×5 kernel matrix** $K$ and add noise:  
$$  
K = \begin{bmatrix}  
k(\mathbf{x}_1, \mathbf{x}_1) & \cdots & k(\mathbf{x}_1, \mathbf{x}_5) \\  
\vdots & \ddots & \vdots \\  
k(\mathbf{x}_5, \mathbf{x}_1) & \cdots & k(\mathbf{x}_5, \mathbf{x}_5)  
\end{bmatrix} + \sigma_n^2 I  
$$  

**Example calculations**:  
- $k(\mathbf{x}_1, \mathbf{x}_1) = \exp(0) = 1$  
- $k(\mathbf{x}_1, \mathbf{x}_2) = \exp\left(-\frac{(32-64)^2}{2 \cdot 100^2} - \frac{(0.001-0.01)^2}{2 \cdot 0.1^2}\right) \approx \exp(-0.0512 - 0.0405) \approx 0.913$  
- $k(\mathbf{x}_2, \mathbf{x}_3) = \exp\left(-\frac{(64-128)^2}{2 \cdot 100^2} - \frac{(0.01-0.005)^2}{2 \cdot 0.1^2}\right) \approx \exp(-0.2048 - 0.00125) \approx 0.824$  

**Resulting kernel matrix** (approximate):  
$$  
K \approx \begin{bmatrix}  
1.01 & 0.913 & 0.741 & 0.607 & 0.913 \\  
0.913 & 1.01 & 0.824 & 0.607 & 0.741 \\  
0.741 & 0.824 & 1.01 & 0.607 & 0.741 \\  
0.607 & 0.607 & 0.607 & 1.01 & 0.607 \\  
0.913 & 0.741 & 0.741 & 0.607 & 1.01  
\end{bmatrix}  
$$  


##### **4. Predict at a New Point**  
Let’s predict accuracy for $$\mathbf{x}_* = [96, 0.007]$$.  

##### **Step 1: Compute Kernel Vector**  
$$  
\mathbf{k}_* = \begin{bmatrix}  
k(\mathbf{x}_*, \mathbf{x}_1) \\  
k(\mathbf{x}_*, \mathbf{x}_2) \\  
k(\mathbf{x}_*, \mathbf{x}_3) \\  
k(\mathbf{x}_*, \mathbf{x}_4) \\  
k(\mathbf{x}_*, \mathbf{x}_5)  
\end{bmatrix}  
$$  
Calculate each term:  
- $$k(\mathbf{x}_*, \mathbf{x}_1) \approx \exp\left(-\frac{(96-32)^2}{20000} - \frac{(0.007-0.001)^2}{0.02}\right) \approx \exp(-0.2048 - 0.0018) \approx 0.824$$  
- $$k(\mathbf{x}_*, \mathbf{x}_2) \approx \exp\left(-\frac{(96-64)^2}{20000} - \frac{(0.007-0.01)^2}{0.02}\right) \approx \exp(-0.0512 - 0.00045) \approx 0.951$$  
- $$k(\mathbf{x}_*, \mathbf{x}_3) \approx \exp\left(-\frac{(96-128)^2}{20000} - \frac{(0.007-0.005)^2}{0.02}\right) \approx \exp(-0.0512 - 0.0002) \approx 0.950$$  
- $$k(\mathbf{x}_*, \mathbf{x}_4) \approx \exp\left(-\frac{(96-256)^2}{20000} - \frac{(0.007-0.02)^2}{0.02}\right) \approx \exp(-1.28 - 0.00845) \approx 0.277$$  
- $$k(\mathbf{x}_*, \mathbf{x}_5) \approx \exp\left(-\frac{(96-512)^2}{20000} - \frac{(0.007-0.001)^2}{0.02}\right) \approx \exp(-8.66 - 0.0018) \approx 0.0002$$  

$$  
\mathbf{k}_* \approx \begin{bmatrix} 0.824 \\ 0.951 \\ 0.950 \\ 0.277 \\ 0.0002 \end{bmatrix}  
$$  

##### **Step 2: Compute Posterior Mean**  
Solve $$K^{-1} \mathbf{y}$$ (using Gaussian elimination or matrix inversion):  
$$  
K^{-1} \approx \begin{bmatrix}  
1.45 & -0.55 & -0.55 & 0.09 & -0.44 \\  
-0.55 & 1.45 & -0.55 & 0.09 & -0.44 \\  
-0.55 & -0.55 & 1.45 & 0.09 & -0.44 \\  
0.09 & 0.09 & 0.09 & 1.09 & -0.36 \\  
-0.44 & -0.44 & -0.44 & -0.36 & 1.68  
\end{bmatrix}  
$$  
*(Note: This is an illustrative approximation; exact inversion would require precise calculations.)*  

Then:  
$  
\mu_* = \mathbf{k}_*^\top K^{-1} \mathbf{y} \approx [0.824, 0.951, 0.950, 0.277, 0.0002] \cdot \begin{bmatrix} 1.45 \cdot 0.82 + \cdots \\ \vdots \\ \end{bmatrix}  
$  
For simplicity, assume $$K^{-1} \mathbf{y} \approx [0.82, 0.85, 0.88, 0.83, 0.80]$$ (approximate due to inversion complexity):  
$  
\mu_* \approx 0.824 \cdot 0.82 + 0.951 \cdot 0.85 + 0.950 \cdot 0.88 + 0.277 \cdot 0.83 + 0.0002 \cdot 0.80 \approx 0.86  
$  

##### **Step 3: Compute Posterior Variance**  
$  
\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top K^{-1} \mathbf{k}_*  
$  
$ 
k(\mathbf{x}_*, \mathbf{x}_*) = 1 + \sigma_n^2 = 1.01  
$  
$  
\sigma_*^2 \approx 1.01 - [0.824, 0.951, 0.950, 0.277, 0.0002] \cdot K^{-1} \mathbf{k}_* \approx 0.02  
$  

---

##### **5. Final Prediction**  
For $\mathbf{x}_* = [96, 0.007]$:  
$  
\text{Accuracy} = 0.86 \pm 0.14 \quad (1 \sigma)  
$  

---

##### **Key Notes**  
1. **Simplifications**:  
   - The inverse $K^{-1}$ is approximated for illustration.  
   - Real-world applications require numerical tools for precise inversion.  
2. **Kernel Choice**:  
   - Adjust $l_1$ and $l_2$ based on hyperparameter scales (see [Search Result 6] for learning rate/batch size relationships).  
3. **Noise**: $\sigma_n^2 = 0.01$ assumes moderate measurement uncertainty.  


