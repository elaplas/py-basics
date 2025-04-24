### Random variable
- Its value depends on a random event
- All possible outcomes in a sample space 
- e.g. the value of the random variable "coin" with possible outcomes {Head, Tail}
---
### Joint random variable and joint probability distribution
- Joint random variables are two or more random variables considered together on the same probability space, where their **simultaneous behavior** is described by a joint probability distribution. This allows us to study the relationship between the variables, such as how likely they are to take certain values at the same time, and whether knowing the value of one affects the probability of the other.

- Joint random variables are simply random variables analyzed together, with their combined behavior described by a joint probability distribution.

##### Formal definition

- **Discrete case**:  
  For two discrete random variables X and Y, the **joint probability mass function** (joint pmf) is:
  $$P(X = x, Y = y) $$ This gives the probability that X takes value x and Y takes value y simultaneously.

- **Continuous case**:  
  For two continuous random variables, the **joint probability density function** (joint pdf) is:
  $$f_{X,Y}(x, y)$$
  This function describes the relative likelihood that X is near x and Y is near y.

- **Generalization**:  
  The concept extends to any number of random variables, not just two. For example, for three variables X, Y, Z, the joint distribution is $$P(X = x, Y = y, Z = z)$$.

##### Properties

- The joint distribution encodes both the **marginal distributions** (the individual distributions of each variable) and the **conditional distributions** (how one variable behaves given the value of another).
- **Marginal probability**: it is the probability of a single event occurring, independent of other events.

- **conditional probability**: it is the probability that an event occur s given that another specific event has already occurred

- For discrete variables, the joint pmf sums to 1 over all possible value pairs:
  $$
  \sum_x \sum_y P(X = x, Y = y) = 1
  $$
- For continuous variables, the joint pdf integrates to 1 over all possible values:
  $$
  \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x, y) dx\,dy = 1
  $$

##### Examples

- The grade you receive on an exam and the number of hours you slept the night before.
- The daily returns of two different stocks in a portfolio.
- The box office revenue of a movie and its critical rating[3].

##### Why are joint random variables important?
---
Studying joint random variables lets us:
- Analyze **dependencies** and **correlations** between variables.
- Compute probabilities of combined events.
- Understand how the outcome of one variable affects another (conditional probability).
---
### Joint probability example

Each of two urns contains two red balls and one blue ball, and one ball is randomly selected from each urn, with the two draws independent of each other. Let A and B be discrete random variables associated with the outcomes of the draw from the first urn and second urn respectively. The probability (marginal probability) of drawing a red ball from either of the urns is ⁠2/3⁠, and the probability of drawing a blue ball is ⁠1/3⁠.

|            | A = Red             | A = Blue            | P(B)           |
|------------|---------------------|---------------------|----------------|
| **B = Red**  | (2/3)(2/3) = 4/9    | (1/3)(2/3) = 2/9    | 4/9 + 2/9 = 2/3 |
| **B = Blue** | (2/3)(1/3) = 2/9    | (1/3)(1/3) = 1/9    | 2/9 + 1/9 = 1/3 |
| **P(A)**     | 4/9 + 2/9 = 2/3     | 2/9 + 1/9 = 1/3     |                |

**Interpretation:**
- Each cell shows the joint probability P(A, B).
- Row and column sums give the marginal probabilities P(A) and P(B).
---
### Standard deviation
- The standard variation of a random variable is the variation of the random variable about its mean   
- The standard deviation of a random variable is the square root of its variance $$ \sigma(\mathbf{x}) = \sqrt{Var(\mathbf{x})} $$ 
---
### Variance
- The variance of a random variable is the mean of the square of the deviations of the samples substracted from their mean $$ Var(\mathbf{x}) = \mathbb{E} \left[ (\mathbf{x} - \mu)^2 \right] $$ 
---
### Covariance
- When two or more random variables are defined on a probability space, it is useful to describe how they vary together; that is, it is useful to measure the relationship between the variables. A common measure of the relationship between two random variables is the covariance. Covariance is a measure of linear relationship between the random variables. If the relationship between the random variables is nonlinear, the covariance might not be sensitive to the relationship, which means, it does not relate the correlation between two variables. 

$$ Cov(\mathbf{x}, \mathbf{y}) = \mathbb{E} \left[ (\mathbf{x} - \mu_x) (\mathbf{y} - \mu_y)  \right] $$ 

- In a covariance matrix the elements on the diagonal represent the variances and the values on the upper and lower triangles are the covariances/correlations
---
### Correlation

- The correlation just scales the covariance by the product of the standard deviation of each variable. Consequently, the correlation is a dimensionless quantity that can be used to compare the linear relationships between pairs of variables in different units. 
---
### Normal distribution 
- It is a probability distribution descri bing the probability distribution of a continuous random variable x with an special exponential function:

$$ \mathbf{f}(\mathbf{x}) = \frac{1}{\sqrt{2\pi\sigma^2}} \mathbf{e}^{-\frac{(\mathbf{x}-\mu)^2}{2\sigma^2}} $$

- The narrower the normal distribution, the more probable is the drawn sample from such a distribution to be close to the mean

- The wider the normal distribution, the less probable is the drawn sample from such a distribution to be close to the mean.

- The drawn sample from a normal distribution is most probably in the range of standard deviation. 

- The drawn sample might fall anywhere in the infinity because the normal distribution is extended to infinity.

- If the probability of the mean of a normal distribution is 0.9, it means that if we draw a sample from such a distribution, it will fall around the mean value with the probability of 0.9.

---
### Multi-variate normal distribution

- It is used to describe the joint probability distribution of any set of random variables, each of which clusters around a mean value. We need the mean vector and the covariance matrix of random variables to be able to calculate the joint normal distribution.

- For two random variable the joint normal distribution looks like a hill or mountain. If we draw a sample pair from such a ditribution, the values of the sample pairs are most likely close to the pick of the mountain. 

- A *finite-dimensional* probability distribution over a vector of random variables. For a random vector $$\mathbf{X} = (X_1, ..., X_n)$$, the joint distribution is:  
$$
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}),
$$  
where:  
- $\boldsymbol{\mu}$: Mean vector  
- $\boldsymbol{\Sigma}$: Covariance matrix  

---

### Bayesian rule/estimator

- Bayes' theorem (alternatively Bayes' law or Bayes' rule, after Thomas Bayes) gives a mathematical rule for inverting conditional probabilities, allowing one to find the probability of a cause given its effect.


- It combines prior knowledge with observed data to update beliefs.  

### **Mathematical Statement**  
For events $A$ and $B$:  
$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$  
- **$P(A|B)$**: Posterior probability of $A$ given $B$.  
- **$P(B|A)$**: Likelihood of observing $B$ if $A$ is true (known as measurement).  
- **$P(A)$**: Prior probability of $A$.  
- **$P(B)$**: Marginal probability of $B$.  

#### **Derivation**  
1. **Conditional probability**:  
   $P(A|B) = \frac{P(A \cap B)}{P(B)}$  
2. **Joint probability**:  
   $P(A \cap B) = P(B|A) \cdot P(A)$  
3. **Substitute**:  
   $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$  
  
