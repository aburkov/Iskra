# Building the GRPO Algorithm from First Principles  

Mathematical reasoning is hard for language models. It demands both precise quantitative reasoning and careful multi-step problem-solving. A trained model can improve at math through **reinforcement learning** (RL). In this tutorial, we build **Group Relative Policy Optimization** (GRPO) from scratch—an RL algorithm designed to train language models on tasks with exact answers, such as coding, math, and logic. GRPO takes advantage of reasoning steps, where the model generates a chain of thought before answering, which has been shown to improve accuracy.  

GRPO is a simplified version of **Proximal Policy Optimization** (PPO), an algorithm commonly used in **reinforcement learning from human feedback** (RLHF) to align models with user preferences. Unlike PPO, GRPO doesn't use a **critic model**, which simplifies training and reduces memory use. Instead of learning from a critic's estimated values, GRPO estimates baselines from groups of sampled outputs. This works well for math problems since rewards can be assigned by comparing the generated answer to the ground truth through string matching—eliminating the need for human feedback.

Below is an alternative derivation—drawing inspiration from the DeepSeek‐Math paper—that shows how we replace the raw reward signal $r(q,o)$ with a composite learning signal that combines an advantage term and a KL‐penalty term. This derivation explains how GRPO’s per–token loss emerges.

## 1. The Standard Policy Gradient Objective

We begin with the usual objective for a policy $\pi_\theta(o \mid q)$:

$$
J(\theta) = 𝔼_{q\sim P(Q),\, o\sim \pi_\theta(O\mid q)}\Bigl[ r(q,o)\Bigr]
$$

Taking the gradient and applying the log‐derivative trick, we have:

$$
\nabla_\theta J(\theta) = 𝔼_{q,o}\Bigl[ r(q,o)\,\nabla_\theta \log \pi_\theta(o\mid q) \Bigr].
$$
This is the classic REINFORCE update.


## 2. Replacing the Raw Reward with an Advantage

In practice, we do not use the raw reward directly. Instead, we compute an advantage function $A(q,o)$ that measures how much better a given output is compared to a baseline (often obtained by averaging rewards over multiple completions for the same prompt). That is, we set:

$$
r(q,o) \quad\longrightarrow\quad A(q,o).
$$

Thus, the gradient becomes:

$$
\nabla_\theta J(\theta) = 𝔼_{q,o}\Bigl[ A(q,o)\,\nabla_\theta \log \pi_\theta(o\mid q) \Bigr].
$$

## 3. Adding a KL‐Divergence Regularizer

To prevent the policy from straying too far from a reference (or baseline) model $\pi_{\text{ref}}$, we introduce a KL‐divergence penalty. In GRPO, this penalty is incorporated at the per–token level. For a token $o_t$ in a given sequence, let:

$$
\Delta_t = \log \pi_\theta(o_t \mid q, o_{<t}) - \log \pi_{\text{ref}}(o_t \mid q, o_{<t}).
$$

An **unbiased estimator** of the KL divergence is:

$$
f(\Delta_t) = \exp\bigl(-\Delta_t\bigr) + \Delta_t - 1,
$$

which can also be written as

$$
f(\Delta_t) = \exp\Bigl(\log \pi_{\text{ref}}(o_t\mid q,o_{<t}) - \log \pi_\theta(o_t\mid q,o_{<t})\Bigr) - \Bigl(\log \pi_{\text{ref}}(o_t\mid q,o_{<t}) - \log \pi_\theta(o_t\mid q,o_{<t})\Bigr) - 1.
$$

This term is zero when the current and reference log–probabilities match and increases when they diverge.

We then penalize the advantage by a factor $\beta$ (which controls the strength of the regularization), and define a composite learning signal per token:

$$
\tilde{r}(q,o_t) = A(q,o_t) - \beta\, f(\Delta_t).
$$

Thus, the gradient update becomes

$$
\nabla_\theta J(\theta) \approx 𝔼_{q,o}\Bigl[ \tilde{r}(q,o_t)\,\nabla_\theta \log \pi_\theta(o_t\mid q,o_{<t})\Bigr].
$$

## 4. Incorporating Variance Reduction via Importance Weighting

In practice—especially when sampling multiple completions per prompt—the advantage $A(q,o)$ is computed in a group–relative manner. Moreover, to prevent gradients from “leaking” through the baseline, we re–weight the advantage with an importance factor:
$$
w(o_t) = \exp\Bigl(\log \pi_\theta(o_t\mid q,o_{<t}) - \text{stop\_grad}\bigl(\log \pi_\theta(o_t\mid q,o_{<t})\bigr)\Bigr),
$$
which effectively equals 1 but ensures that the gradient only flows through the numerator. Then the per–token contribution becomes:
$$
\text{Loss}(o_t) = -\Bigl( w(o_t)\,A(q,o_t) - \beta\, f(\Delta_t) \Bigr).
$$

---

## 5. Final GRPO Loss Expression

Averaging over all tokens in the generated output $o$ (and across multiple samples per prompt), the GRPO loss is given by:
$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \frac{1}{|o^{(i)}|}\sum_{t=1}^{|o^{(i)}|} \Biggl( \exp\Bigl(\log \pi_\theta(o^{(i)}_t\mid q,o^{(i)}_{<t}) - \text{stop\_grad}\bigl(\log \pi_\theta(o^{(i)}_t\mid q,o^{(i)}_{<t})\bigr)\Bigr) A(q,o^{(i)}_t) - \beta\, f\bigl(\Delta^{(i)}_t\bigr) \Biggr),
$$
where $N$ is the number of sampled outputs (often grouped by prompt). Minimizing this loss (or, equivalently, maximizing its negative) updates the policy to favor completions with high relative advantage while penalizing those that diverge too far from the reference model.

---

### 6. Summary

- **Advantage Substitution:** Replace $r(q,o)$ with the advantage $A(q,o)$ computed over groups of completions.
- **KL Penalty:** For each token, add a penalty
  $$
  f(\Delta_t) = \exp\bigl(\log \pi_{\text{ref}}(o_t\mid q,o_{<t}) - \log \pi_\theta(o_t\mid q,o_{<t})\bigr) - \Bigl(\log \pi_{\text{ref}}(o_t\mid q,o_{<t}) - \log \pi_\theta(o_t\mid q,o_{<t})\Bigr) - 1,
  $$
  scaled by $\beta$.
- **Importance Weighting:** Ensure gradients only flow through the current model’s parameters by using a detached baseline in the weighting.
- **Final Loss:** The resulting GRPO loss per token is
  $$
  \ell_t = -\Bigl( \underbrace{\exp\Bigl(\log \pi_\theta(o_t\mid q,o_{<t}) - \text{stop\_grad}(\log \pi_\theta(o_t\mid q,o_{<t}))\Bigr) A(q,o_t)}_{\text{Policy Gradient Term}} - \beta\, \underbrace{\Bigl(\exp\bigl(\log \pi_{\text{ref}}(o_t\mid q,o_{<t}) - \log \pi_\theta(o_t\mid q,o_{<t})\bigr) - \bigl(\log \pi_{\text{ref}}(o_t\mid q,o_{<t}) - \log \pi_\theta(o_t\mid q,o_{<t})\bigr) - 1\Bigr)}_{\text{KL Penalty}} \Bigr).
  $$
  
This derivation shows how—in GRPO—we replace the original reward $r(q,o)$ with a learning signal
$$
\tilde{r}(q,o) = A(q,o) - \beta\, \Bigl(\exp\bigl(\log \pi_{\text{ref}}(o\mid q) - \log \pi_\theta(o\mid q)\bigr) - \bigl(\log \pi_{\text{ref}}(o\mid q) - \log \pi_\theta(o\mid q)\bigr) - 1\Bigr),
$$
thereby ensuring that outputs which both yield high (relative) advantage and remain close to the reference model are reinforced. This is the core idea behind the GRPO loss used in our implementation.

--- 

This complete derivation—mirroring the style and level of detail in the DeepSeek‐Math article—shows the motivation and mathematical steps leading to the composite learning signal used in GRPO.
