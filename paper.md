# Optimizer

## ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION https://arxiv.org/pdf/1412.6980

### Algorithm:

![image-20250208124206828](./assets/image-20250208124206828.png)

### Novelty:

In comparison with adagrad and RMSProp, Adam introduces the concept of second moment of the gradient to stabilize the gradient update.

### Explanation:

**First Moment:**

$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
 &=  (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i
\end{align*}
$$
Let $g^m_t$ denote the exponential moving average of the gradient at time $t$.
$$
\begin{align*}
E[m_t] & \approx E[g^m_t] (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \\
& \approx E[g^m_t] (1-\beta_1) (\frac{1-\beta_1^t}{1-\beta_1}) & \text{(Geometric Series)} \\ 
& \approx E[g^m_t] (1-\beta_1^t)
\end{align*}
$$

So, by dividing $m_t$ by $1-\beta_1^t$, we can get the unbiased estimate of the first moment $E[\hat{m_t}] \approx E[g^m_t]$.

**Second Moment:**

$$
\begin{align*}
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
&= (1-\beta_2) \sum_{i=1}^{t} \beta_2^{t-i} g_i^2
\end{align*}
$$

$$
\begin{align*}
E[v_t] & \approx E[(g^m_t)^2] (1-\beta_2) \sum_{i=1}^{t} \beta_2^{t-i} \\
&\approx E[(g^m_t)^2] (1-\beta_2) (\frac{1-\beta_2^t}{1-\beta_2}) & \text{(Geometric Series)} \\
& \approx E[(g^m_t)^2] (1-\beta_2^t)
\end{align*}
$$

So, by dividing $v_t$ by $1-\beta_2^t$, we can get the unbiased estimate of the second moment $E[\hat{v_t}] \approx E[(g^m_t)^2]$.

**Why $\frac{\hat{m_t}}{\sqrt{\hat{v_t}}}$ works?**

- It makes the gradient update $\alpha\frac{\hat{m_t}}{\sqrt{\hat{v_t}}}$ sit at a bounded region (estimated) [-1, 1] because $\frac{\hat{m_t}}{\sqrt{\hat{v_t}}} \approx \frac{E[g^m_t]}{\sqrt{E[(g^m_t)^2]}} \lessapprox 1$

- Second moment represents the variance of the gradient. When the variance is large, it means it is uncertain about the true direction of the gradient to the local optimum. So, we want it to take a smaller step in case of overshooting from the local minima.





