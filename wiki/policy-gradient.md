## Policy Gradient

In policy gradient methods for reinforcement learning, we want to adjust the parameters of our policy $\pi_\theta$ so that outputs (which are the complete text completions that the language model generates) that lead to high rewards become more likely. The key steps are as follows:

### 1. Starting from the Objective

We want to maximize the expected reward that the agent accumulates by generating completions and obtaining rewards:

$$
J(\theta) = 𝔼_{q \sim P(Q), o \sim \pi_\theta(O \mid q)}[r(q, o)]
$$

Here, $J(\theta)$ is the **objective** we want to maximize, $r(q, o)$ is the reward obtained by generating the complete output $o$ in response to a prompt $q$. The notation $q \sim P(Q)$ means that we take expectation with respect to the distribution of possible queries, where each query $q$ is sampled from the probability distribution $P(Q)$. Similarly, each complete output $o$ is sampled from our policy (parameterized by $\theta$) which defines a probability distribution $\pi_\theta(O|q)$ over possible complete outputs given the query.

### 2. Deriving the Gradient of the Objective

To maximize the objective, we calculate its gradient. Let's derive the gradient of the objective $J(\theta)$ with respect to the policy parameters $\theta$.

We can rewrite the objective as follows:

$$
J(\theta) = 𝔼_{q\sim P(q)}\Bigl[𝔼_{o \sim \pi_\theta(\cdot \mid q)}[ r(q,o)]\Bigr]
$$

Here, $P(q)$, the distribution over prompts, is independent of $\theta$, because we randomly sample prompts from the training dataset. Because $P(q)$ does not depend on $\theta$, we can bring the gradient inside the outer expectation:

$$
\nabla_\theta J(\theta) = 𝔼_{q\sim P(q)}\Bigl[\nabla_\theta 𝔼_{o \sim \pi_\theta(\cdot \mid q)}[r(q,o)]\Bigr]
$$

Inside the inner expectation, we then differentiate with respect to $\theta$:

$$
𝔼_{o \sim \pi_\theta(\cdot \mid q)}[r(q,o)] = \sum_{o} \pi_\theta(o \mid q)r(q,o)
$$

Taking the gradient with respect to $\theta$ and applying [the log-derivative trick](log_derivative_trick.md):

$$
\nabla_\theta \pi_\theta(o \mid q) = \pi_\theta(o \mid q) \nabla_\theta \log \pi_\theta(o \mid q),
$$

we obtain:

$$
\nabla_\theta 𝔼_{o \sim \pi_\theta(\cdot \mid q)}[ r(q,o) ] = \sum_{o} r(q,o)\pi_\theta(o \mid q)\nabla_\theta \log \pi_\theta(o \mid q)
$$

This can be written in expectation form as:

$$
\nabla_\theta 𝔼_{o \sim \pi_\theta(\cdot \mid q)}[r(q,o)] = 𝔼_{o \sim \pi_\theta(\cdot \mid q)}\Bigl[r(q,o)\nabla_\theta \log \pi_\theta(o \mid q)\Bigr]
$$

Now, plugging this back into the full gradient, we have:

$$
\nabla_\theta J(\theta) = 𝔼_{q\sim P(q)}\Bigl[𝔼_{o \sim \pi_\theta(\cdot \mid q)}\bigl[r(q,o) \, \nabla_\theta \log \pi_\theta(o \mid q) \bigr]\Bigr]
$$

In the policy gradient update, for each token $o_{i,t}$ in a generated sequence (with its corresponding prompt $q$), we update the parameters by moving in the direction:

$$
\nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})
$$

??????????

This gradient is then scaled by a factor (the "gradient coefficient") which typically includes the advantage (or reward signal) and any additional corrections (such as KL divergence gradients). Multiplying by the reward signal ensures that tokens leading to higher rewards are reinforced.

So, in summary:
- **We don't lose the prompt distribution** because the gradient is taken under the expectation over $q$; that part remains throughout the derivation.
- **The log probability term** appears because differentiating the probability $\pi_\theta(o \mid q)$ via the log-derivative trick naturally produces $\nabla_\theta \log \pi_\theta(o \mid q)$.
- **Multiplying by the reward (or advantage)** adjusts the update magnitude to favor actions that yield higher rewards.

This formal approach preserves the expectation over prompts while using the log-probability gradient to direct the parameter updates.


$$
\nabla_\theta J(\theta) = \nabla_\theta 𝔼_{o \sim \pi_\theta(\cdot \mid q)} [r(q, o)].
$$

Because the expectation is over the policy, we can use the "log-derivative trick" (or REINFORCE trick), which tells us that:

$$
\nabla_\theta \pi_\theta(o \mid q) = \pi_\theta(o \mid q)\, \nabla_\theta \log \pi_\theta(o \mid q).
$$
This allows us to write:
$$
\nabla_\theta J(\theta) = 𝔼_{o \sim \pi_\theta(\cdot \mid q)}\left[ r(q, o) \, \nabla_\theta \log \pi_\theta(o \mid q) \right].
$$
The key point is that the gradient of the log of the policy appears naturally as the result of differentiating the probability distribution with respect to its parameters. This is where the term $\nabla_\theta \log \pi_\theta(o \mid q)$ comes from.

---

### 3. Why Multiply by the Reward?

Multiplying the gradient of the log probability by the reward (or an advantage function) has a very intuitive meaning:

- **Direction of Improvement:**  
  The term $\nabla_\theta \log \pi_\theta(o \mid q)$ tells us how a small change in the parameters $\theta$ will affect the log probability of generating $ o $. If an action $ o $ yields a high reward, we want to increase its probability. By multiplying by $ r(q, o) $ (or a more refined advantage $ \hat{A} $), we weight the gradient update so that actions with higher rewards are reinforced.
  
- **Scaling Updates Appropriately:**  
  If the reward is high, the product $ r(q, o)\, \nabla_\theta \log \pi_\theta(o \mid q) $ is large, causing a larger update that increases the probability of $ o $. Conversely, if the reward is low (or negative), the update will push the policy away from generating $ o $.

- **Variance Reduction (via Advantage):**  
  In practice, instead of using the raw reward, we often use an advantage $ \hat{A}(q, o) $ which measures how much better an action is compared to a baseline. This helps reduce variance. The update then becomes:
  $$
  \nabla_\theta J(\theta) \approx 𝔼_{o \sim \pi_\theta(\cdot \mid q)}\left[ \hat{A}(q, o) \, \nabla_\theta \log \pi_\theta(o \mid q) \right].
  $$

Thus, multiplying by the reward (or advantage) ensures that the update moves the policy parameters in the direction that increases the likelihood of actions that yield higher rewards and decreases the likelihood of actions that yield lower rewards.

---

### 4. Summary in the Context of GRPO

In the GRPO framework, the per-token update is of the form:
$$
\text{Gradient Update} \propto \Bigl(\hat{A}_{i,t} + \text{(KL penalty term)}\Bigr) \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}).
$$
Here:
- $ \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) $ comes from the log-derivative trick.
- $ \hat{A}_{i,t} $ is the advantage signal computed (for example, by normalizing rewards across multiple generated sequences).
- The KL penalty term (derived by differentiating the KL divergence between the current and reference policies) ensures the new policy does not stray too far from the reference.

By scaling the gradient of the log probability with the advantage (and KL correction), the update effectively reinforces the probability of tokens that contributed to higher rewards, which is precisely why we multiply the gradient by the reward signal.

---

This formal derivation explains both where the log of the policy arises in the policy gradient update and why it is multiplied by the reward (or advantage) to improve the policy.
