# Reinforcement Learning

## Trust Region Policy Optimization https://arxiv.org/pdf/1502.05477

### Novelty:

- Optimize the policy with the constraint on the parameters of the KL divergence between the old and new policy. By smartly translating the optimization problem into a quadratic programming problem, the optimization then becomes maximizing the expected return of the policy while keeping the KL divergence between the old and new policy as low as possible. 
- Conjugate Gradient Method is used to solve the quadratic programming problem. Because the derived quadratic function of KL divergence is a matrix vector product with the Hessian of KL divergence, which is a symmetric matrix. Thus, the conjugate gradient method is a good choice in comparison to the steepest descent method.
- Fisher Vector Product (Hessian of KL divergence is a Fisher Information Matrix) is used to efficiently compute the $Hv$ without explicitly computing the Hessian matrix $H$, which is a $n^2$ matrix. Finding the first order derivative of KL divergence is enough to compute the Fisher Vector Product.

### Preliminaries:

**Expected Return of Policy $\tilde{\pi}$:**

$$
\begin{align*}
\eta(\tilde{\pi}) &= \eta(\pi) + \mathbb{E}_{\tau \sim \tilde{\pi}}[\sum_{t=0}^{\infty} \gamma^t A_{\pi}(s_t, a_t)]\\
&= \eta(\pi) + \sum_{t=0}^{\infty} \sum_{s \in \mathcal{S}} P(s_t = s|\tilde{\pi}) \sum_{a \in \mathcal{A}} \gamma^t \tilde{\pi}(a|s) A_{\pi}(s, a)\\
&= \eta(\pi) + \sum_{s \in \mathcal{S}} \rho_{\tilde{\pi}}(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a|s) A_{\pi}(s, a)\\
&\text{where } \rho_{\tilde{\pi}}(s) = \sum_{t=0}^{\infty} P(s_t = s|\tilde{\pi}) \gamma^t \text{ is the discounted visitation frequency of state $s$ under policy $\tilde{\pi}$}
\end{align*}
$$

**Surrogate Expected Return of Policy $\tilde{\pi}$ with respect to policy $\pi$:**

In practice, it would be infeasible to estimate or sample the visitation frequency of state $s$ under the updated policy $\tilde{\pi}$. Instead, we use the old discounted visitation frequency $\rho_{\pi}$ of state $s$ under the old policy $\pi$.

$$
\begin{align*}
L_{\pi}(\tilde{\pi}) &=\eta(\pi) + \sum_{s \in \mathcal{S}} \rho_{\pi}(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a|s) A_{\pi}(s, a)\\
\end{align*}
$$


**Policy Improvement Bound:**

$$
\begin{align*}
 \eta(\tilde{\pi}) \geq L_{\pi}(\tilde{\pi})  - \frac{4\epsilon \gamma}{(1-\gamma)^2} \alpha^2, \text{ where } \alpha = D_{TV}^{max}(\pi, \tilde{\pi}) \quad \epsilon = \max_{s,a} \left| A_\pi(s,a) \right|
\end{align*}
$$

### Architecture:

**Objective Function:**

$$
\begin{align*}
& \text{maximize}_{\theta} 
\quad 
\nabla_{\theta} L_{\theta_{\text{old}}}(\theta)\Big|_{\theta = \theta_{\text{old}}}
\,\cdot\,
(\theta - \theta_{\text{old}}) 
\\
& \text{subject to} 
\quad 
\frac{1}{2}\,(\theta_{\text{old}} - \theta)^\top \, A(\theta_{\text{old}})\, (\theta_{\text{old}} - \theta) 
\;\;\le\;\; 
\delta,
\end{align*}
$$

$$
\begin{align*}
\hspace{15mm} \text{where}\quad
A(\theta_{\text{old}})_{ij} 
\;=\;
\frac{\partial}{\partial \theta_i}\,\frac{\partial}{\partial \theta_j} 
\;\mathbb{E}_{s \sim \rho_\pi}
\Big[
D_{\mathrm{KL}}\bigl(\pi(\cdot\mid s, \theta_{\text{old}})\,\big\|\,\pi(\cdot\mid s, \theta)\bigr)
\Big]
\Bigg|_{\theta = \theta_{\text{old}}}
\end{align*}
$$

**Intuition:**

- Monotonic Improvement:By improving the surrogate objective, we are guaranteed to improve the true objective. (see the proof section)

- Taylor Expansion of KL divergence: We obtain the Hessian of the KL divergence by expanding the KL divergence to second order. (See the below section)

- Symmetric property of Hessian of KL divergence: $A$ is the second order derivative of the KL divergence, which is a symmetric matrix (Hessian of KL divergence). And the symmetric property is  useful for doing conjugate gradient. (See the proof section)
- Fisher Vector Product: $A$ is the Fisher information matrix, which is the Hessian of the KL divergence, and it gives us the advanatages of not fully computing the Hessian matrix.
- Quadratic Form for Steepest Descent/Conjugate Gradient: A steepest descent formulation aims to find the minimum of a quadratic function $f(x)=\frac{1}{2}x^T A x - b^T x + c$. The minimum of this function is found by solving $f'(x) = Ax-b=0$. Since the fisher vector $A$ is symmetric, we can use conjugate gradient to solve the linear system $Ax=b$ efficiently.

- Search Direction and Step Size: Based on the constraint, we need to restrict the KL divergence between the old and new policy to be less than $\delta$. However, in solving the $Ax=b$, there is no involvement of the constraint $\delta$. So, the author proposed a non-trivial method to calculate the step size based on the calculated search direction $x$ to enforce the constraint. (See the next section)

Thus, by updating $\theta$ with the search direction multiplied by a step size, we can ensure that the surrogate objective function is improved while the KL divergence between the old and new policy is less than $\delta$.

**Calculation of Step Size Given Search Direction:**

Suppose we have computed the search direction $s=A^{-1}g$, where $g$ is the gradient of the surrogate objective function and $A$ is the Hessian of the KL divergence. We need to find the step size $\beta$ to update $\theta=\theta_{\text{old}}+\beta s$ such that the quadratic constraint of KL divergence is satisfied.

$$
\begin{align*}
\frac{1}{2}\,(\theta_{\text{old}} - \theta)^\top \, A\, (\theta_{\text{old}} - \theta) &= \delta \\
\frac{1}{2}\,(\beta s)^\top \, A\, (\beta s) &= \delta \\
\frac{1}{2}\,\beta^2 s^\top \, A\, s &= \delta \\
\beta^2 &= \frac{2\delta}{s^\top \, A\, s} \\
\beta &= \sqrt{\frac{2\delta}{s^\top \, A\, s}}
\end{align*}
$$

**Taylor Expansion of KL divergence:**

Let $D_{KL}(\mu(\theta))$ to denote the $D_{KL}(\pi_{\theta_{\text{old}}} \big\|\pi_\theta)$ between the old and new policy.

By Taylor expansion, we have:

$$
\begin{align*}
D_{KL}(\mu(\theta)) &\approx D_{KL}(\mu(\theta_{\text{old}})) +  D'_{KL}(\mu(\theta_{\text{old}})) (\theta - \theta_{\text{old}}) + \frac{1}{2} (\theta - \theta_{\text{old}})^T  H (\theta - \theta_{\text{old}})\\
&= 0 + 0 + \frac{1}{2} (\theta - \theta_{\text{old}})^T  H (\theta - \theta_{\text{old}}) \\
&= \frac{1}{2} (\theta - \theta_{\text{old}})^T  H (\theta - \theta_{\text{old}})
\end{align*}
$$

Because:

$$
\begin{align*}
 D'_{KL}(\mu(\theta_{\text{old}})) &= \frac{\partial  D_{KL}(\mu(\theta_{\text{old}}))}{\partial \mu(\theta_{\text{old}})}  \frac{\partial \mu(\theta_{\text{old}})}{\partial \theta}\\
 &=0
\end{align*}
$$

$$
\begin{align*}
 D''_{KL}(\mu(\theta_{\text{old}})) &= \frac{\partial^2  D_{KL_{ab}}(\mu(\theta_{\text{old}}))}{\partial \mu_a \partial \mu_b}  \frac{\partial \mu_a(\theta_{\text{old}})}{\partial \theta_i}\frac{\partial \mu_b(\theta_{\text{old}})}{\partial \theta_j} + \frac{\partial  D_{KL_{a}}(\mu(\theta_{\text{old}}))}{\partial \mu_a}  \frac{\partial^2 \mu_a(\theta_{\text{old}})}{\partial \theta_i \partial \theta_j}\\
&= \frac{\partial^2  D_{KL_{ab}}(\mu(\theta_{\text{old}}))}{\partial \mu_a \partial \mu_b}  \frac{\partial \mu_a(\theta_{\text{old}})}{\partial \theta_i}\frac{\partial \mu_b(\theta_{\text{old}})}{\partial \theta_j} + 0\\
&= H_{\mu(\theta_{\text{old}})}
\end{align*}
$$

**Fisher Information Matrix:**

Fisher Vector Product is defined as:
$$
F(\theta)\,v \;=\; \mathbb{E}_{x \sim p_{\theta}}\!\Bigl[ \nabla_{\theta}\log p_{\theta}(x)\;\nabla_{\theta}\log p_{\theta}(x)^{\top} \Bigr]\;v. \
$$

This means that we can first compute the dot product of $\mathbb{E}_{x \sim p_{\theta}}[\nabla_{\theta}\log p_{\theta}(x)^T]$ and $v$, and then multiply the result by $\mathbb{E}_{x \sim p_{\theta}}[\nabla_{\theta}\log p_{\theta}(x)]$. It is efficent because it only takes $O(n)$ time to compute the dot product of $\mathbb{E}_{x \sim p_{\theta}}[\nabla_{\theta}\log p_{\theta}(x)^T]$ and $v$, and $O(n^2)$ time to compute the dot product of $\mathbb{E}_{x \sim p_{\theta}}[\nabla_{\theta}\log p_{\theta}(x)]$ and $v$.

We show that $H = F(\theta)$ is the Fisher Information Matrix by showing that $H=D''_{Kl}=\mathbb{E}_{\pi_{\theta_{old}}}[ \nabla_{\theta}^2 \log \pi_{\theta}(a|s)]$
$$
\begin{align*}
D'_{Kl} &= \nabla_{\theta} D_{KL}(\mu(\theta)) = \nabla_{\theta} \int \pi_{\theta_{old}}(a|s) \log \frac{\pi_{\theta_{old}}(a|s)}{\pi_{\theta}(a|s)} da \\
&= \int \pi_{\theta_{old}}(a|s) \nabla_{\theta} \log \frac{\pi_{\theta_{old}}(a|s)}{\pi_{\theta}(a|s)} da \\
&= \int \pi_{\theta_{old}}(a|s) \nabla_{\theta} \log \pi_{\theta_{old}}(a|s) da - \int \pi_{\theta_{old}}(a|s) \nabla_{\theta} \log \pi_{\theta}(a|s) da\\
&= 0 - \mathbb{E}_{\pi_{\theta_{old}}}[ \nabla_{\theta} \log \pi_{\theta}(a|s)] \\
&= - \mathbb{E}_{\pi_{\theta_{old}}}[ \nabla_{\theta} \log \pi_{\theta}(a|s)]
\end{align*}
$$

$$
\begin{align*}
H=D''_{Kl} &= \nabla_{\theta} D'_{Kl} = \nabla_{\theta} \mathbb{E}_{\pi_{\theta_{old}}}[ \nabla_{\theta} \log \pi_{\theta}(a|s)] \\
&= \mathbb{E}_{\pi_{\theta_{old}}}[ \nabla_{\theta}^2 \log \pi_{\theta}(a|s)]
\end{align*}
$$

### Pseudo Code:

**Fisher Vector Product:**

```python
def compute_fisher_vector_product(v):
    # 1. Compute the average KL divergence over the given states and actions.
    #    This KL divergence is between the current policy and a reference (often the old policy).
    kl = compute_average_KL_divergence()

    # 2. Compute the gradient of the KL divergence with respect to the policy parameters.
    #    Enable the creation of a computation graph to allow second-order derivative computation.
    grad_kl = gradient(kl, policy_parameters, create_graph=True)

    # 3. Compute the directional derivative (dot product) of the gradient in the direction of v.
    grad_dot_v = dot_product(grad_kl, v)

    # 4. Compute the Hessian–vector product by differentiating the dot product with respect to policy parameters.
    hessian_vector_product = gradient(grad_dot_v, policy_parameters)

    # 5. Add a damping term for numerical stability.
    fisher_vector_product = hessian_vector_product + damping * v

    return fisher_vector_product
```

**Conjugate Gradient (Search Direction):**

```python
def conjugate_gradient(A_function=computeFisherVectorProduct, b, max_iterations, tolerance):
    # A_function: a function that returns A @ v (matrix-vector product) given a vector v
    # b: the right-hand side vector
    # max_iterations: maximum number of iterations to run
    # tolerance: convergence threshold for the residual
    
    x = zeros_like(b)           # Initial guess (often zero)
    r = b - A_function(x)       # Residual: r = b - Ax
    p = r.copy()                # Initial direction vector
    rsold = dot(r, r)           # Squared norm of the residual
    
    for i in range(max_iterations):
        Ap = A_function(states, actions, p)      # Compute A * p using the provided function
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p       # Update the solution estimate
        r = r - alpha * Ap      # Update the residual
        
        rsnew = dot(r, r)       # New squared norm of the residual
        if sqrt(rsnew) < tolerance:
            break               # Convergence criteria met
        
        beta = rsnew / rsold    # Compute the new conjugate direction factor
        p = r + beta * p        # Update the direction vector
        rsold = rsnew           # Prepare for the next iteration
    
    return x

```

**Step Size Calculation:**

```python
def compute_step_size(A_function=compute_fisher_vector_product, s, delta, max_iterations, tolerance):
    # A_function: a function that returns A @ v (matrix-vector product) given a vector v
    # s: the search direction
    # delta: the constraint

    # Compute the step size
    # beta = sqrt(2 * delta / sAs))
    beta = sqrt(2 * delta / dot(s, A_function(s)))
    
```
### Proof:

**Show $\eta(\tilde{\pi}) = \eta(\pi) + \mathbb{E}_{\tau \sim \tilde{\pi}}[\sum_{t=0}^{\infty} \gamma^t A_{\pi}(s_t, a_t)]$**

$$
\begin{align*}
A_{\pi}(s, a) &= Q_{\pi}(s, a) - V_{\pi}(s)\\
&= \mathbb{E}_{s'\sim P(s'|s, a)}[r(s) + \gamma V_{\pi}(s')] - V_{\pi}(s)\\

\end{align*}
$$


$$
\begin{align*}
\mathbb{E}_{\tau \sim \tilde{\pi}}[\sum_{t=0}^{\infty} \gamma^t A_{\pi}(s_t, a_t)] &= \mathbb{E}_{\tau \sim \tilde{\pi}}[\sum_{t=0}^{\infty} \gamma^t (r(s_t) + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t))]\\

&= \mathbb{E}_{\tau \sim \tilde{\pi}}[\sum_{t=0}^{\infty} \gamma^t r(s_t)] + \mathbb{E}_{\tau \sim \tilde{\pi}}[\sum_{t=0}^{\infty} \gamma^{t+1}  V_{\pi}(s_{t+1})- \gamma^t V_{\pi}(s_t)]\\

&= \mathbb{E}_{\tau \sim \tilde{\pi}}[\sum_{t=0}^{\infty} \gamma^t r(s_t)] - \mathbb{E}_{\tau \sim \tilde{\pi}}[V_{\pi}(s_0)]\\

&=\eta(\tilde{\pi}) - V_{\pi}(s_0)\\
&= \eta(\tilde{\pi}) - \eta(\pi)
\end{align*}

$$

**Policy Improvement Bound:**

By definition, the surrogate expected return is:

$$
\begin{align*}
L_{\pi}(\tilde{\pi}) &= \eta(\pi) + \sum_{s \in \mathcal{S}} \rho_{\pi}(s) \sum_{a \in \mathcal{A}} \tilde{\pi}(a|s) A_{\pi}(s, a)\\
&= \eta(\pi) + \mathbb{E}_{s_t \sim \pi}[\sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{a_t \sim \tilde{\pi}(a_t|s_t)}[A(s_t, a_t)]]\\
\end{align*}

$$

Definition 1: let $(\pi, \tilde{\pi})$ be a $\alpha$-coupled distribution such that for $a\sim \pi, a'\sim \tilde{\pi}$, $P(a\not=a'|s) \leq \alpha$.


$$
\begin{align*}  
\mathbb{E}_{\tilde{a} \sim \tilde{\pi}}[A_{\pi}(s, \tilde{a})] &= \mathbb{E}_{\tilde{a} \sim \tilde{\pi}}[A_{\pi}(s, \tilde{a})] - \mathbb{E}_{a \sim \pi} [A_{\pi}(s, a)] \quad \text{by definition of } \mathbb{E}_{a \sim \pi}[A_{\pi}(s, a)] = 0\\
&\leq P(\tilde{a}\not=a|s)\max_{a, \tilde{a}}\left|A_{\pi}(s, \tilde{a}) -A_{\pi}(s, a)\right| \\
&\leq 2\alpha \max_{s,a}\left|A_{\pi}(s, a)\right| \\
\end{align*}
$$

Next, we need to figure out the bound of the difference between $\mathbb{E}_{s_t \sim \pi}[ \mathbb{E}_{a_t \sim \tilde{\pi}}[A_{\pi}(s_t, a_t)]]$ and $\mathbb{E}_{s_t \sim \tilde{\pi}}[\mathbb{E}_{a_t \sim \tilde{\pi}}[A_{\pi}(s_t, a_t)]]$. Starting from here, we denote $A(s_t)$ as $\mathbb{E}_{a_t \sim \tilde{\pi}}[A_{\pi}(s_t, a_t)]$. So the difference is $\mathbb{E}_{s_t \sim \pi}[A(s_t)] - \mathbb{E}_{s_t \sim \tilde{\pi}}[A(s_t)]$.

And let $n_t$ denote the number of times $a_i$ disagrees with $\tilde{a}_i$ before time $t$. And we use can use it to estimate the chance that trajectory $\tau$ agrees with  $\tilde{\tau}$ by $P(n_t=0) = (1-\alpha)^{t}$, and the chance that $\tau$ disagrees with $\tilde{\tau}$ by $P(n_t>0) = 1 -P(n_t=0) = 1-(1-\alpha)^{t}$.

Thus,

$$
\begin{align*}
\left | \mathbb{E}_{s_t\sim\pi}[A(s_t)] - \mathbb{E}_{s_t\sim\tilde{\pi}}[A(s_t)] \right| &= \left | \mathbb{E}_{s_t\sim\pi|n_t=0}[A(s_t)] + \mathbb{E}_{s_t\sim\pi|n_t>0}[A(s_t)] - \mathbb{E}_{s_t\sim\tilde{\pi}|n_t=0}[A(s_t)] - \mathbb{E}_{s_t\sim\tilde{\pi}|n_t>0}[A(s_t)] \right| \\
&= \left | \mathbb{E}_{s_t\sim\pi|n_t=0}[A(s_t)] - \mathbb{E}_{s_t\sim\tilde{\pi}|n_t=0}[A(s_t)] \right| + \left | \mathbb{E}_{s_t\sim\pi|n_t>0}[A(s_t)] - \mathbb{E}_{s_t\sim\tilde{\pi}|n_t>0}[A(s_t)] \right| \\
&=\left| \mathbb{E}_{s_t\sim\pi|n_t>0}[A(s_t)] - \mathbb{E}_{s_t\sim\tilde{\pi}|n_t>0}[A(s_t)] \right|\\
&\leq P(n_t>0) \cdot 2 \max_{s_t}\left|A(s_t) \right|\\
&\leq (1-(1-\alpha)^t) \cdot 2 \cdot 2\alpha \max_{s,a}\left|A(s, a)\right|\\
&= 4\alpha (1-(1-\alpha)^t) \max_{s,a}\left|A(s, a)\right|\\
\end{align*}
$$

Now we can compute the bound of the difference between $L_{\pi}(\tilde{\pi})$ and $\eta(\tilde{\pi})$:

$$
\begin{align*}
\sum_{t=0}^{\infty} \gamma^t \left| \mathbb{E}_{s_t\sim\pi}[A(s_t)] - \mathbb{E}_{s_t\sim\tilde{\pi}}[A(s_t)] \right| &\leq \sum_{t=0}^{\infty} \gamma^t 4\alpha (1-(1-\alpha)^t) \max_{s,a}\left|A(s, a)\right|\\
&= 4\alpha \max_{s,a}\left|A(s, a)\right| \sum_{t=0}^{\infty} \gamma^t (1-(1-\alpha)^t)\\
&= 4\alpha \max_{s,a}\left|A(s, a)\right| \sum_{t=0}^{\infty} \gamma^t - \gamma^t(1-\alpha)^t\\
&= 4\alpha \max_{s,a}\left|A(s, a)\right| (\frac{1}{1-\gamma} -  \frac{1}{1-\gamma(1-\alpha)})\\
&= \frac{4\alpha^2\gamma \max_{s,a}\left|A(s, a)\right|}{(1-\gamma)(1-\gamma(1-\alpha))}\\
&\leq \frac{4\alpha^2\gamma}{(1-\gamma)^2} \max_{s,a}\left|A(s, a)\right|\\
\end{align*}
$$





**Monotonic Improvement over Surrogate Objective:**

  We know that $\eta({\pi_{i+1}})$ is bounded by $L_{\pi_i}(\pi_{i+1})$ with the policy improvement bound. Now, we need to show that if $L_{\pi_i}(\pi_{i+1})$ for $i > 0$ is a monotonic increasing sequence, then $\eta({\pi_{i}})$ for $i > 0$ is also a monotonic increasing sequence.

  $$
  \begin{align*}
  \eta(\pi_{i+1}) &\geq L_{\pi_i}(\pi_{i+1}) \\
  \eta(\pi_{i+1}) - L_{\pi_i}(\pi_{i}) &\geq L_{\pi_i}(\pi_{i+1})   - L_{\pi_i}(\pi_{i})\\
  \eta(\pi_{i+1}) - \eta(\pi_i) &\geq L_{\pi_i}(\pi_{i+1}) - L_{\pi_i}(\pi_{i})
  \end{align*}
  $$

  Based on this derivation, we show that  $\eta(\pi_{i+1})$ is a monotonic increasing sequence if we keep on improving the surrogate objective.