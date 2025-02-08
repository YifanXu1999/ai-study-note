# Optimizer

## ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION https://arxiv.org/pdf/1412.6980

### Algorithm:

![image-20250208124206828](./assets/image-20250208124206828.png)

### Novelty:

In comparison with adagrad and RMSProp, Adam introduces the concept of second moment of the gradient to stabilize the gradient update. Theoretically and empirically, it is successful at smoothing the overshooting problem.

### Explanation:

**First Moment:**

$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
 &=  (1-\beta_1) \sum_{i=0}^{t} \beta_1^{t-i} g_i
\end{align*}
$$

Let $l = (1-\beta_1)\sum_{i=0}^{t} \beta_1^{t-i} ( g_i - g_t)$, then

$$
\begin{align*}
E[m_t] & = E[g_t] (1-\beta_1) \sum_{i=0}^{t} \beta_1^{t-i} + l\\
& = E[g_t] (1-\beta_1) (\frac{1-\beta_1^t}{1-\beta_1})  + l& \text{(Geometric Series)} \\ 
& = E[g_t] (1-\beta_1^t) + l
\end{align*}
$$


If $g$ is stationary, $l$ = 0. Even if $g$ is non-stationary, the author argued that $l$ is relatively small. Due to the exponential decay factor $\beta_1$, the difference of the early timestep is almost negligible.

So, by dividing $m_t$ by $1-\beta_1^t$, we can get the unbiased estimate of the first moment $E[\hat{m_t}] \approx E[g_t]$.

**Second Moment:**

$$
\begin{align*}
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
&= (1-\beta_2) \sum_{i=0}^{t} \beta_2^{t-i} g_i^2
\end{align*}
$$

$$
\begin{align*}
E[v_t] & = E[g_t^2] (1-\beta_2) \sum_{i=0}^{t} \beta_2^{t-i} + l \\
& = E[g_t^2] (1-\beta_2) (\frac{1-\beta_2^t}{1-\beta_2}) + l & \text{(Geometric Series)} \\
& = E[g_t^2] (1-\beta_2^t) + l
\end{align*}
$$

Similar to the first moment, we treat $l$ as a negligible term.

So, by dividing $v_t$ by $1-\beta_2^t$, we can get the unbiased estimate of the second moment $E[\hat{v_t}] \approx E[g_t^2]$.

**Why $\frac{\hat{m_t}}{\sqrt{\hat{v_t}}}$ works?**

- It makes the gradient update $\alpha\frac{\hat{m_t}}{\sqrt{\hat{v_t}}}$ sit at a bounded region (estimated) [-1, 1] because $\frac{\hat{m_t}}{\sqrt{\hat{v_t}}} \approx \frac{E[g_t]}{\sqrt{E[g_t^2]}} \lessapprox 1$

- Second moment represents the variance of the gradient. When the variance is large, it means it is uncertain about the true direction of the gradient to the local optimum. So, we want it to take a smaller step in case of overshooting from the local minima.





