# Statistical Methodology & Mathematical Formulation

**Project**: BDS-34 Capstone Platform  
**Title**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments  

---

## 1. Non-Contractual Customer Behavior Paradigm

In retail e-commerce (such as the UCI Online Retail II giftware catalog), customers transact in a **continuous time, non-contractual** setting. Customers make purchases intermittently and can drop out at any point without explicit notification. 

To model this stochastic process, we implement the seminal **BG/NBD (Beta-Geometric / Negative Binomial Distribution)** model formulated by Fader, Hardie, and Lee (2005).

---

## 2. Mathematical Formulations

### 2.1 BG/NBD Repeat Purchase Model
1. While an individual customer is "alive" (active), the number of transactions $x$ in a time period of length $t$ follows a Poisson distribution with transaction rate $\lambda$:
   $$P(X(t) = x \mid \lambda) = \frac{(\lambda t)^x e^{-\lambda t}}{x!}$$
2. Heterogeneity in $\lambda$ across customers follows a Gamma distribution with shape $r$ and scale $\alpha$:
   $$f(\lambda \mid r, \alpha) = \frac{\alpha^r \lambda^{r-1} e^{-\alpha \lambda}}{\Gamma(r)}$$
3. After any transaction, a customer drops out with unobserved probability $p$. Dropout opportunity occurs immediately after each transaction.
4. Heterogeneity in dropout probability $p$ across customers follows a Beta distribution with shape parameters $a$ and $b$:
   $$f(p \mid a, b) = \frac{p^{a-1} (1-p)^{b-1}}{B(a, b)}$$
5. Given a customer's observed history of repeat transactions $x$, recency $t_x$ (time of last repeat purchase), and observation tenure $T$, the probability that the customer is currently active is:
   $$P(\text{Alive} \mid x, t_x, T) = \left( 1 + \frac{a}{b + x - 1} \left( \frac{\alpha + T}{\alpha + t_x} \right)^{r + x} \right)^{-1} \quad \text{for } x > 0$$
6. The conditional expected number of repeat purchases over a future horizon $t$ is:
   $$E[Y(t) \mid x, t_x, T] = \frac{a + b + x - 1}{a - 1} \left[ 1 - \left( \frac{\alpha + T}{\alpha + T + t} \right)^{r + x} {}_2F_1\left(r + x, b + x; a + b + x - 1; \frac{t}{\alpha + T + t}\right) \right] P(\text{Alive})$$

### 2.2 Gamma-Gamma Monetary Spend Model
1. The monetary value $z_{i1}, z_{i2}, \dots, z_{ix}$ of individual transactions for customer $i$ is distributed Gamma with shape $p$ and scale $\nu$:
   $$f(z \mid p, \nu) = \frac{\nu^p z^{p-1} e^{-\nu z}}{\Gamma(p)}$$
2. Heterogeneity in transaction scale $\nu$ across customers follows a Gamma distribution with shape $q$ and scale $\gamma$:
   $$f(\nu \mid q, \gamma) = \frac{\gamma^q \nu^{q-1} e^{-\gamma \nu}}{\Gamma(q)}$$
3. Conditional on observed repeat frequency $x$ and average repeat basket spend $m_x$:
   $$E[M \mid p, q, \gamma, x, m_x] = \frac{q - 1}{p x + q - 1} \cdot \frac{\gamma p}{q - 1} + \frac{p x}{p x + q - 1} \cdot m_x$$
   This is an empirical Bayes weighted average between the customer's observed average spend $m_x$ and the population mean spend $\frac{\gamma p}{q - 1}$.

### 2.3 Discounted CLV & Bootstrap Uncertainty Intervals
Discounted expected customer lifetime value over horizon $H$ with monthly discount factor $d$ is:
$$\text{CLV}_i(H) = E[Y_i(H)] \cdot E[M_i] \cdot \left(1 + d\right)^{-H / 60}$$

To avoid reporting misleading point estimates, the system simulates 50 posterior draws from the joint distribution:
$$\tilde{Y}_i \sim \text{Poisson}(E[Y_i(H)]), \quad \tilde{M}_i \sim \text{Gamma}\left(k=4, \theta = \frac{E[M_i]}{4}\right)$$
The 80% prediction interval is determined empirically:
$$\text{Interval}_{80\%} = \left[ Q_{0.10}(\tilde{\text{CLV}}_i), Q_{0.90}(\tilde{\text{CLV}}_i) \right]$$
