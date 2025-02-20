Let the true KL divergence between the reference policy $\pi_{\text{ref}}$ and the current policy $\pi_\theta$ be defined as

$$
D_{KL}(\pi_{\text{ref}}\parallel \pi_\theta) = \sum_{o} \pi_{\text{ref}}(o)\log\frac{\pi_{\text{ref}}(o)}{\pi_\theta(o)}.
$$

Now, define the random variable

$$
X = \frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)}
$$

with $o$ sampled according to $\pi_{\text{ref}}$. Notice that

$$
\mathbb{E}_{o\sim\pi_{\text{ref}}}[X] = \sum_{o} \pi_{\text{ref}}(o)\frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} = \sum_{o} \pi_\theta(o) = 1.
$$

It is known that for any positive random variable $X$ with $\mathbb{E}[X]=1$, the function

$$
h(X) = X - \log X - 1
$$

satisfies

$$
\mathbb{E}_{o\sim\pi_{\text{ref}}}\bigl[h(X)\bigr] = D_{KL}(\pi_{\text{ref}}\parallel \pi_\theta).
$$

Substitute $X = \frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)}$ into $h(X)$:

$$
h\!\Bigl(\frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)}\Bigr) = \frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} - \log\frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} - 1.
$$

Taking the expectation over $o \sim \pi_{\text{ref}}$, we obtain

$$
\mathbb{E}_{o\sim\pi_{\text{ref}}}\!\left[\frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} - \log\frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} - 1\right] = D_{KL}(\pi_{\text{ref}}\parallel \pi_\theta).
$$

This demonstrates that the term

$$
\frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} - \log\frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} - 1
$$

is an unbiased estimator of $D_{KL}(\pi_{\text{ref}}\parallel \pi_\theta)$.

*(Note: Writing it equivalently as*

$$
\frac{\pi_{\text{ref}}(o)}{\pi_\theta(o)} + \log\frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)} - 1
$$

*simply flips the fraction inside the logarithm and the sign accordingly.)*
