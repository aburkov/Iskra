## Policy Gradient Methods and the REINFORCE algorithm

In policy gradient methods for reinforcement learning, we want to adjust the parameters of our policy $\pi_\theta$ (the model that returns a distribution over actions for a given state of the environment) so that the actions that lead to high rewards become more likely. While rewards in reinforcement learning can come, depending on the environemnt, after each action executed by the agent, we are specifically interested in a more realistic scenario, where a reward is only obtained from the environemnt after the agent reaches some final state, which can be the state in which the task is completed (in which case the reward is usually high) or failed (low reward). In the case of training a neural language model, where indiviadual actions are predicted tokens, such a final state is usually a full generated text in response to a certain input prompt. We denote by $q$ the input query (which corresponds to the initial environment state) and by $o$, the entire generated sequence when the query $q$ is the input.

The key steps in policy gradient are as follows:

### 1. Starting from the Objective

We want to maximize the expected reward that the agent accumulates by generating completions and obtaining rewards:

$$
J(\theta) = 𝔼_{q \sim P(Q), o \sim \pi_\theta(O \mid q)}[r(q, o)]
$$

Here, $J(\theta)$ is the **objective** we want to maximize, $r(q, o)$ is the reward obtained by generating the complete output $o$ in response to a prompt $q$. The notation $q \sim P(Q)$ means that we take expectation with respect to the distribution of possible queries, where each query $q$ is sampled from the probability distribution $P(Q)$. Similarly, each complete output $o$ is sampled from our policy (parameterized by $\theta$) which defines a probability distribution $\pi_\theta(O|q)$ over possible complete outputs given the query.

To give an example, let our input be "How much is 1+1?". This is our $q$. Assume that tokenization is made by words. The probability that our policy $\pi_\theta$ produces the output sequence "The answer is 2." that we denote as $\pi_\theta(\text{The answer is 2.} \mid \text{How much is 1+1?})$ is given by $\pi_\theta(\text{The}\mid \text{How much is 1+1?})$ multiplied by $\pi_\theta(\text{answer} \mid \text{How much is 1+1? The})$ multiplied by $\pi_\theta(\text{is} \mid \text{How much is 1+1? The answer})$, $\ldots$, multiplied by $\pi_\theta(\text{.} \mid \text{How much is 1+1? The answer is 2})$. 

Given a policy $\pi_\theta$, the probability to generate a complete sequence $o$ is given by the product of probabilities 

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
\nabla_\theta J(\theta) = 𝔼_{q\sim P(q)}\Bigl[𝔼_{o \sim \pi_\theta(\cdot \mid q)}\bigl[r(q,o)\nabla_\theta \log \pi_\theta(o \mid q) \bigr]\Bigr]
$$

### 3. Replacing the Reward $r(q, o)$ with an Arbitrary Term

In many practical settings, the reward $r(q, o)$ is not the only signal we care about. For example, we might introduce:

1. **Advantage Functions**: Instead of using the raw reward $r(q,o)$, it is common to use an **advantage function** $A(q,o)$ that estimates how much better a particular action (or completion) is relative to some baseline. The idea is to replace $r(q,o)$ with $A(q,o)$ so that our gradient becomes

$$
\nabla_\theta J(\theta) = 𝔼_{q\sim P(q)}\Bigl[𝔼_{o \sim \pi_\theta(\cdot \mid q)}\bigl[A(q,o)\nabla_\theta \log \pi_\theta(o \mid q) \bigr]\Bigr]
$$

2. **KL Divergence Terms**: When we want to prevent the policy from deviating too far from a reference (or previous) policy, a KL divergence penalty can be added. For instance, if we have a penalty term $\beta\text{KL}[\pi_\theta(\cdot|q) \,\|\, \pi_{\text{ref}}(\cdot|q)]$ where $\beta$ is a scaling factor, we can add this term to our advantage. In practice, this means our "reward" term in the gradient can become

$$
\tilde{r}(q,o) = A(q,o) + \beta\text{KL}[\pi_\theta(\cdot|q) \,\|\, \pi_{\text{ref}}(\cdot|q)]
$$

The gradient now reads

$$
\nabla_\theta J(\theta) = 𝔼_{q\sim P(q)}\Bigl[𝔼_{o \sim \pi_\theta(\cdot \mid q)}\Bigl[\tilde{r}(q,o)\nabla_\theta \log \pi_\theta(o \mid q) \Bigr]\Bigr].
$$

In both cases, the substitution doesn't change the fundamental structure of the policy gradient update; rather, it modifies the learning signal so that it can be more stable or incorporate additional constraints that guide learning.

### 4. Moving from Expectations to Sample Averages

In theory, our gradient expression is an expectation over all queries $q$ and completions $o$. However, in practice, we don’t have access to the full distributions. Instead, we can sample a batch of queries $\{q_i\}$ from our training set and for each query sample (i.e., use the trained language model to generate) completions $\{o_i\}$ from $\pi_\theta(o|q_i)$.

Thus, the expectation can be approximated by the following **Monte Carlo estimate**:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \tilde{r}(q_i,o_i)\nabla_\theta \log \pi_\theta(o_i \mid q_i),
$$

where $\tilde{r}(q_i,o_i)$ represents the learning signal. This signal could be the original reward $r(q_i, o_i)$, an advantage $A(q_i, o_i)$, or a combination with a KL divergence term as described above.

#### Why is This Useful?

1. **Flexibility**: By substituting $r(q,o)$ with an advantage or an augmented reward, we can incorporate additional information (like a baseline for variance reduction or a regularization penalty) without changing the underlying derivation.

2. **Sample Efficiency**: Sampling queries and completions allows us to approximate the expectation with finite samples. This makes the algorithm computationally tractable and allows us to update the parameters $\theta$ using stochastic gradient ascent.

3. **Empirical Optimization**: In real-world training, we compute the gradient using these sample averages over mini-batches. This Monte Carlo estimate is unbiased and, with enough samples, closely approximates the true gradient of our objective.
