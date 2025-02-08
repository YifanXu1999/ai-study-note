
# Optimization

## Local Minima vs Saddle Point

A local minimum is a point where the function value is less than all nearby points, but may not be the global minimum. A saddle point is a point that is a local minimum in one direction but a local maximum in another direction.

To distinguish between them, we can look at the Taylor series expansion around a critical point (where gradient is zero):

$$
\begin{align*}
f(x) &\approx f(x_0) + \nabla f(x_0)^T(x-x_0) + \frac{1}{2}(x-x_0)^T H(x_0)(x-x_0)  \\
&\approx f(x_0) + \frac{1}{2}(x-x_0)^T H(x_0)(x-x_0) 
\end{align*}
$$

where H(x_0) is the Hessian matrix at x_0. At a critical point:

- If all eigenvalues of H(x_0) are positive → Local minimum
- If all eigenvalues of H(x_0) are negative → Local maximum  
- If eigenvalues have mixed signs → Saddle point



## Batch Normalization

### Small Batch vs Large Batch

- Calculation Speed Per Batch: Due to the advancement of matrix parallelization, the training speed per batch is almost the same for small and large batch. The performance difference would become noticiable when the batch size became too large.
- Training Speed Per Epoch: Because larger batch size calculation can be almost as fast as small batch, the training speed per epoch depeding how many updates needed to be performed per batch. Thus, larger batch size has a faster training speed per epoch.
- Convergence: Small batch in general has a better convergence. Due to the noise in the gradient, small batch has a higher variance in the gradient. It is less likely to get stuck in a local&sharp minimum.

### Normalization

## Momentum

<img src="./assets/1738948191244.png" alt="1738948191244" style="zoom:50%;" />

In the phyiscal world, momentum is the inertia of an object. In the picture above, the ball will continue to roll down the hill, and the momentum will overcome the slight upward slope and continue to roll down.

### Gradient Descent with Momentum

$$
\begin{align*}
\text{Starting at } \theta^0 \\
\text{Compute Gradient } g^0 \\
\text{Update Movement } m^0 &=  \eta g^0  \\
\text{Update Parameter } \theta^1 &= \theta^0 - m^0 \\
\text{Compute Gradient } g^1 \\
\text{Update Movement } m^1 &=  \eta g^1 - \lambda m^0   \\
\text{Update Parameter } \theta^2 &= \theta^1 - m^1  \\
\end{align*}
$$


## Adaptive Learning Rate

Why need adaptive learning rate?

- (Need for Gradually Decreased Learning Rate) At the final stage of training, we might not need the learning rate to be as large as the initial stage. Because a plausible trajectory of training of loss function goes like at the initial stage, we want the parameters to explore the space more. Thus, parameters need to have a larger learning rate to jump from one area to another. However, at the final stage, most parameters are likely to be stuck at a area such that they are unlikely to get out with more iterations of training. Thus, we want them to converge to the local minima. And for this case, smaller learning rate is preferred to get the result more accurate.
- (Different Learning Rate for Different Parameters) Different parameters have different gradients. For example, parameters with large gradient, they are likely do not need large learning rate to get out of the local minima. Instead, a large learning rate might lead to overshoot and oscillation. 

Thus, we want to introduce a learning rate hyper parameter for adaptive adjustment. And turn the parameter update into:

$$
\theta^{t+1} = \theta^t - \frac{\eta }{\sigma^t} g^t
$$




### AdaGrad

$$
\sigma^t =\sqrt{\frac{1}{t+1}\sum_{i=0}^{t} (g_i)^2}
$$

### RMSProp

$$
\sigma^t = \sqrt{\alpha(\sigma^{t-1})^2 + (1-\alpha)(g^t)^2}
$$

### Adam

<img src="./assets/image-20250208124130344.png" alt="image-20250208124130344" style="zoom:50%;" />


### Learning Rate Scheduling




# Math


## Eigenvalues and Eigenvectors



## Taylor Series

$$
f(x) = \sum_{n} \frac{f^{n}(a)(x-a)^n}{n!}
$$

Simple Proof:

Suppose $f(x)$ is a polynomial of $c_0 + c_1(x-a) + c_2(x-a)^2 + c_3(x-a)^3 + \cdots$

Their derivatives are:

$$
\begin{align*}
f'(x) &= c_1 + 2c_2(x-a) + 3c_3(x-a)^2 + \cdots \\
f''(x) &= 2c_2 + 6c_3(x-a) + 12c_4(x-a)^2 + \cdots \\
f'''(x) &= 6c_3 + 24c_4(x-a) + 60c_5(x-a)^2 + \cdots \\
\end{align*}
$$

Let $x = a$, we get:

$$
\begin{align*}
f(a) &= c_0 \\
f'(a) &= c_1 \\
f''(a) &= 2c_2 \\
f'''(a) &= 6c_3 \\
\end{align*}
$$


Thus, we can get the coefficients of the polynomial by the derivatives at $x = a$. And the function can be rewritten as:

$$
\begin{align*}
f(x) &= f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2 + \frac{f'''(a)}{6}(x-a)^3 + \cdots \\
\end{align*}
$$


# Neural Network

## Feed Forward Neural Network 


# Activation Functions

## Sigmoid

**Function:**

$$
\begin{align*}
f(x) = \frac{1}{1 + e^{-x}}
\end{align*}
\\
$$

**Derivative:**

$$
\begin{align*}
f'(x) &= \frac{e^{-x}}{(1 + e^{-x})^2} \\
&= f(x)(1 - f(x))
\end{align*}
\\
$$


## Tanh

**Define:**

$$
\begin{align*}
 \sigma(x) = sigmoid(x) = \frac{1}{1 + e^{-x}}
\\
\end{align*}
\\
$$

**Function:**

$$
\begin{align*}
f(x) &= 2 \cdot \sigma(2x) - 1  \\
&= \frac{e^x - e^{-x}}{e^x + e^{-x}}
\end{align*}
\\
$$

**Derivative:**

$$
\begin{align*}
f'(x) &= 4 \sigma'(2x) \\
&= 4[\frac{e^{-2x}}{(1 + e^{-2x})^2}] \\
\end{align*}
\\
$$

**Derivative Comparison against sigmoid:**

$$
\begin{array}{|c|c|c|}
\hline
x & \text{sigmoid}'(x) & \text{tanh}'(x) \\
\hline
-10 & 0.000045 & 0.000004 \\
-5 & 0.0067 & 0.0018 \\
-2 & 0.105 & 0.070 \\
-1 & 0.197 & 0.240 \\
0 & 0.250 & 1.000 \\
1 & 0.197 & 0.240 \\
2 & 0.105 & 0.070 \\
5 & 0.0067 & 0.0018 \\
10 & 0.000045 & 0.000004 \\
\hline
\end{array}
$$

**Why tanh is (slightly) better than sigmoid:**

Based on the derivative, we can see that tanh has a steeper gradient around 0, and the range of tanh derivative is larger than sigmoid ([0, 1] vs [0, 0.25]). Despite it has a worse gradient vanishing problem, tanh is still better than sigmoid for most cases.

**Issue of sigmoid and tanh:**

- Vanishing gradient problem (when the input is too large or too small)
- Sensitive to weight initialization (the initial weight has profound impact to the gradient due to its unbalanced gradient distribution)
- Computational Complexity (its derivative function is more complex than ReLU)
- Less Sparse Activation (Every neuron is activated at every layer, which might lead to overfitting)








## ReLU

**Function:**

$$
\begin{align*}
f(x) = \max(0, x)
\end{align*}
\\
$$







