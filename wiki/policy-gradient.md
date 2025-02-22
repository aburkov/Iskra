# Policy Gradient Methods and the REINFORCE Algorithm

In policy gradient methods for reinforcement learning, we want to adjust the parameters $\theta$ of our parametrized policy $\pi_\theta$ (the model that returns a distribution over actions for a given state of the environment) so that the actions that lead to high rewards become more likely.

> This article is part of a series that will serve as building blocks for **The Hundred-Page Reinforcement Learning Book** by Andriy Burkov. The book is still in progress and is expected to be published by late 2025. To stay updated, subscribe to the [newsletter](https://aiweekly.substack.com/).

While rewards in reinforcement learning could come, depending on the environment, after each action executed by the agent, we are specifically interested in a more realistic scenario, where a reward is only obtained from the environment after the agent reaches some final state, which can be the state in which the task is completed (in which case the reward is usually high) or failed (low reward).

In the case of training a neural language model, where individual actions are predicted tokens, such a final state is usually a full generated text in response to a certain input prompt seen as the agent's initial state in the environment. We denote by $q$ the input query (which corresponds to the initial environment state) and by $o$, the entire generated sequence when the query $q$ is the input.

The key steps in policy gradient are as follows:

## Policy Gradient Objective

We want to maximize the expected reward that the agent accumulates by generating completions and obtaining rewards:

$$
J(\theta) = 𝔼_{q \sim P(Q), o \sim \pi_\theta(O \mid q)}[r(q, o)]
$$

Here, $J(\theta)$ is the **objective** we want to maximize, $r(q, o)$ is the reward obtained by generating the complete output $o$ in response to a prompt $q$, $P(Q)$ is the probability distribution over input queries, where each query $q$ belongs to a set $Q$ of all possible inputs; $\pi_\theta(O|q)$ is the policy which, in this case, is used as a probability distribution over full output sequences $o$ belonging to some set $O$ of all possible output sequences. $𝔼$ denotes the **expectation** or the **expected value** of the total reward our agent would accumulate if it executed policy $\pi_\theta$ forever, such that each time it starts from a random state $q$ drawn from $P(Q)$ and draws a completion sequence $o$ from $\pi_\theta(O|q)$.

The expectation in statistics is always with respect to some probability distribution. $P(q)$ is the distribution over input prompts. The notation $q \sim P(Q)$ means that we take expectation with respect to the distribution of possible queries, where each query $q$ is sampled from the probability distribution $P(Q)$. Similarly, each complete output $o$ is sampled from our policy (parameterized by $\theta$) which defines a probability distribution $\pi_\theta(O|q)$ over possible complete outputs given the query.

> The notation $q\sim P(q)$ means "when values $q$ are sampled (or drawn) from the distribution $P$ according to the probability $P(q)$ the distribution $P$ assigns to the output with value equal to $q$."

To give an example, let our input be "How much is 1+1?". This is our $q$. Assume that tokenization is made by words. The probability that our policy $\pi_\theta$ produces the output sequence "The answer is 2." that we denote as $\pi_\theta(\text{The answer is 2.} \mid \text{How much is 1+1?})$ is given by $\pi_\theta(\text{The}\mid \text{How much is 1+1?})$ multiplied by $\pi_\theta(\text{answer} \mid \text{How much is 1+1? The})$ multiplied by $\pi_\theta(\text{is} \mid \text{How much is 1+1? The answer})$, $\ldots$, multiplied by $\pi_\theta(\text{.} \mid \text{How much is 1+1? The answer is 2})$.

> In this case, the agent reached the final state with a high reward, because the answer $2$ is correct. If we could define a perfect reward function for all possible states and actions, we could hope to be able to train a perfect agent that can solve any task perfectly. Unfortunately, defining rewards for actions in states is the hardest part in reinforcement learning. Though, for finetuning language models to solve math, logic, or coding problems, simple rule-based reward functions can be created, as we will see in the chapter on [GRPO](GRPO.md).

## Deriving the Gradient of the Objective

To maximize a function, and our objective is a function of input states, we calculate its gradient. Let's derive the gradient of the objective $J(\theta)$ with respect to the policy parameters $\theta$.

We can rewrite the objective as follows:

$$
J(\theta) = 𝔼_{q\sim P(q)}\Bigl[𝔼_{o \sim \pi_\theta(O \mid q)}[ r(q, o)]\Bigr]
$$

$P(q)$, the distribution over the initial states (input prompts), is independent of $\theta$, because we randomly sample prompts from the training dataset.

Because $P(q)$ does not depend on $\theta$ (so we can treat it as a constant term), by using constant multiple rule of differentiation we can bring the gradient inside the outer expectation:

$$
\nabla_\theta J(\theta) = 𝔼_{q\sim P(q)}\Bigl[\nabla_\theta 𝔼_{o \sim \pi_\theta(O \mid q)}[r(q,o)]\Bigr]
$$

> The **constant multiple rule** of differentiation states that the derivative of a constant multiplied by a function equals the constant times the derivative of the function: $\frac{\partial}{\partial x} [c \cdot f(x)] = c \cdot \frac{\partial}{\partial x} f(x)$, therefore $\nabla c \cdot f(x) = c \cdot \nabla f(x)$.

Inside the inner expectation, we can first express it as a sum over all possible outputs:

$$
𝔼_{o \sim \pi_\theta(O \mid q)}[r(q,o)] = \sum_{o \in O} \pi_\theta(o \mid q)r(q,o)
$$

where $\pi_\theta(o \mid q)$ is the probability that the policy $\pi_\theta(O \mid q)$ assign to output sequence $o$ given the input query $q$.

Taking the gradient with respect to $\theta$ of the sum over all possible outputs:

$$
\nabla_\theta \sum_{o \in O} \pi_\theta(o \mid q)r(q,o) = \sum_{o \in O} r(q,o)\nabla_\theta \pi_\theta(o \mid q)
$$

We can apply [the log-derivative trick](log_derivative_trick.md), which states that for any function $f(\theta)$:

$$
\nabla_\theta f(\theta) = f(\theta)\nabla_\theta \log f(\theta)
$$

This identity follows from the chain rule and the fact that $\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)}$.

Applying this to our policy gradient by substituting $f(\theta) = \pi_\theta(o \mid q)$:

$$
\nabla_\theta \pi_\theta(o \mid q) = \pi_\theta(o \mid q)\nabla_\theta \log \pi_\theta(o \mid q)
$$

Therefore:

$$
\sum_{o \in O} r(q,o)\nabla_\theta \pi_\theta(o \mid q) = \sum_{o} r(q,o)\pi_\theta(o \mid q)\nabla_\theta \log \pi_\theta(o \mid q)
$$

This can be written in expectation form as:

$$
\nabla_\theta 𝔼_{o \sim \pi_\theta(O \mid q)}[r(q,o)] = 𝔼_{o \sim \pi_\theta(O \mid q)}\Bigl[r(q,o)\nabla_\theta \log \pi_\theta(o \mid q)\Bigr]
$$

Now that we have derived the gradient of our objective with respect to the policy parameters, we are ready to explain how we use this result in practice. This leads us to the REINFORCE algorithm, a simple yet powerful method for training policies in reinforcement learning.

## The REINFORCE Algorithm

The REINFORCE algorithm is a **Monte Carlo policy gradient method**. "Monte Carlo" here means that we estimate our expected values by sampling. In our case, rather than summing over all possible outputs (which would be impossible in most practical scenarios), we sample outputs from our policy and use these samples to estimate the expectation. The key observation from our derivation is that the gradient of the objective is:

$$
\nabla_\theta J(\theta) = 𝔼_{q\sim P(q), o \sim \pi_\theta(O \mid q)}\Bigl[r(q,o)\nabla_\theta \log \pi_\theta(o \mid q)\Bigr]
$$

In plain language, this equation tells us:

1. For each query $q$ drawn from our dataset or from some distribution $P(Q)$,
2. For each output sequence $o$ sampled from our policy given $q$,
	- Multiply the reward $r(q,o)$ by the gradient of the log-probability $\nabla_\theta \log \pi_\theta(o \mid q)$,
	- And then average these products over many samples.

This average gives us an **unbiased estimate** of the true gradient. "Unbiased" means that if we were able to average over an infinite number of samples, our estimate would equal the true gradient.

### The Update Rule

Using this gradient estimate, we can update the policy parameters $\theta$ in a way that is similar to other gradient ascent methods. In gradient ascent, we adjust the parameters to increase our objective $J(\theta)$. The update rule is:

$$
\theta \leftarrow \theta + \alpha \hat{g}
$$

where:

- $\alpha$ is the **learning rate**, a positive constant that controls how big each update step is,
- $\hat{g}$ is our estimated gradient, calculated as:

$$
\hat{g} = r(q,o)\nabla_\theta \log \pi_\theta(o \mid q)
$$

Notice that we sample a single query $q$ and a single output $o$ in each update. In practice, we often average this over a mini-batch of samples to make the update more stable.

### The REINFORCE Algorithm Step-by-Step

Let’s break down the algorithm into clear steps:

0. **Initialize a policy**: Our policy $\pi_\theta$ is a neural network of the design of our choice with trainable parameters $\theta$. It can be an entirely untrained network, or, if we are training a language model, it's usually a pretrained and supervised-finetuned language model.

1. **Sample a query**: Draw a query $q$ from the distribution $P(Q)$ (this is usually our training dataset of input-output examples.

2. **Generate an output**: Sample an output sequence $o$ from our policy $\pi_\theta(O \mid q)$. Remember that this is done token by token, and the probability of generating the full sequence is the product of the probabilities of each token given the previous tokens.

3. **Compute the Reward**: Obtain the reward $r(q, o)$ from the environment. In our case, this reward is given only at the end of the sequence, based on how well the complete output $o$ meets the desired criteria.

4. **Calculate the Gradient**: Compute the gradient of the log-probability of the generated output with respect to the policy parameters:
   
$$
\nabla_\theta \log \pi_\theta(o \mid q)
$$

   This step involves backpropagation through the neural network that defines the policy.

6. **Scale by the reward**: Multiply the gradient by the reward $r(q,o)$. This scaling makes it so that outputs with higher rewards have a larger influence on the update:

$$
r(q,o) \nabla_\theta \log \pi_\theta(o \mid q)
$$

7. **Update the Parameters**:  
   Adjust the policy parameters in the direction of the gradient:

$$
\theta \leftarrow \theta + \alpha \, r(q,o) \nabla_\theta \log \pi_\theta(o \mid q)
$$

   In practice, you might collect several samples before updating, so you would average the gradient estimates over a mini-batch.

### Multiplying the Gradient by the Reward

In the basic REINFORCE setup where the reward is only provided at the end of the sequence, the same reward is applied to all tokens in the sequence. Here's what happens:

1. **Log-probability as a sum:** Remember that the probability of a full sequence is the product of the probabilities of each token:

$$
\pi_\theta(o \mid q) = \prod_{t=1}^{T} \pi_\theta(a_t \mid q, a_1, \ldots, a_{t-1})
$$
   
   where $a_t$ denotes a specific action taken (token generated) at time step $t$. Taking the log, we have:
   
$$
\log \pi_\theta(o \mid q) = \sum_{t=1}^{T} \log \pi_\theta(a_t \mid q, a_1, \ldots, a_{t-1})
$$
   
   The sum rule of differentiation tells us that the gradient of the log-probability is:
   
$$
\nabla_\theta \log \pi_\theta(o \mid q) = \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid q, a_1, \ldots, a_{t-1})
$$

2. **Multiplying by the reward:** We then multiply this sum by the reward $r(q, o)$ received at the end:

$$
r(q, o) \nabla_\theta \log \pi_\theta(o \mid q) = r(q, o) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid q, a_1, \ldots, a_{t-1})
$$

   So effectively, the reward $r(q, o)$ scales the gradient for each token in the sequence. Each token's contribution to the overall gradient is increased (or decreased) proportionally by the same reward.

While we don't distribute different rewards to different tokens in this simple case, you can think of it as the entire sequence receiving one reward, and every action (or token) in that sequence is "credited" or "blamed" equally based on that final reward.

### REINFORCE in Python

Here's a simple Python implementation of the algorithm:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW

policy, data_reader = get_policy_and_data()
optimizer = AdamW(policy.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in data_reader:
        optimizer.zero_grad()
        outputs, log_probs = policy(batch.queries)
        rewards = compute_reward(batch, outputs)
        loss = - (rewards * log_probs.sum(dim=1)).mean() ➊
        loss.backward()
        optimizer.step()

```

In this code, `get_policy_and_data` is assumed to provide both the policy network (which outputs actions and log probabilities) and the data reader that yields batches of examples. We also assume there's a reward function `compute_reward` that takes generated sequences and returns the reward values for them. For example, if the output sequence is code, the reward function might execute it and return 1 if there code runs and 0 when it fails to run.

Here's what's going on in line ➊:

- Normally, we use gradient ascent to maximize the expected reward. However, optimizers in PyTorch perform gradient descent. By taking the negative of our objective, we turn the maximization problem into a minimization problem.
- The log probabilities are returned per token in the sequence. We sum these along the sequence dimension `(dim=1)` to obtain the log probability for the entire sequence. This sum represents the log of the product of the probabilities of each token in the sequence, since $\log(a \cdot b) = \log(a) + \log(b)$.
- These sequence-level log probabilities are then multiplied by their corresponding rewards element-wise.
- Finally, we take the mean over the mini-batch to calculate the average loss. This ensures that the magnitude of the gradient does not depend on the batch size, leading to a more stable training process.

## From Rewards to Advantages

In the basic REINFORCE formulation, the policy gradient update is expressed as

$$
\nabla_\theta J(\theta) = 𝔼_{q\sim P(q),\, o\sim \pi_\theta(O \mid q)}\Bigl[r(q,o) \, \nabla_\theta \log \pi_\theta(o \mid q)\Bigr],
$$

where $r(q,o)$ denotes the reward obtained after generating the complete output $o$ in response to the query $q$.

Using the raw reward $r(q,o)$ directly can result in gradient estimates with high variance. An improved strategy refines this update by comparing the received reward with an expected reward for the query. This comparison can enabled by considering two functions. The first is the **action-value function** $Q(q,o)$, which represents the expected reward when generating the specific output $o$ in response to $q$. The second is the **value function** $V(q)$, which estimates the expected reward for the query $q$ over all possible outputs, serving as a **baseline**. In many implementations of policy gradient methods that incorporate value estimation, $V(q)$ is defined as

$$
V(q) = 𝔼_{o\sim\pi_\theta(O \mid q)}\bigl[Q(q,o)\bigr]
$$

The difference between these two functions,

$$
A(q,o) = Q(q,o) - V(q),
$$

is called the **advantage**. The advantage indicates how much better or worse the generated output $o$ performed relative to the average expectation for the query $q$.

Replacing the reward $r(q,o)$ with the advantage $A(q,o)$ in the gradient update yields

$$
\nabla_\theta J(\theta) \approx 𝔼_{q\sim P(q),\, o\sim \pi_\theta(O \mid q)}\Bigl[A(q,o) \, \nabla_\theta \log \pi_\theta(o \mid q)\Bigr].
$$

When the baseline $V(q)$ is independent of the output $o$, the expected gradient remains unchanged. This is because

$$
𝔼_{o\sim\pi_\theta(O \mid q)}\Bigl[V(q) \nabla_\theta \log \pi_\theta(o \mid q)\Bigr] = V(q) \cdot 𝔼_{o\sim\pi_\theta(O \mid q)}\Bigl[\nabla_\theta \log \pi_\theta(o \mid q)\Bigr] = V(q) \cdot 0 = 0
$$

so subtracting $V(q)$ does not alter the expected gradient when averaged over the policy’s outputs. For this property to hold, no special constraints on $Q(q,o)$ are needed beyond its standard definition as the expected future reward when generating output $o$ in response to query $q$. The baseline $V(q)$ is defined as the expectation of $Q(q,o)$ under the policy:

$$
V(q) = 𝔼_{o\sim\pi_\theta(O \mid q)}[Q(q,o)]
$$

This definition of $V(q)$ ensures that using $A(q,o)$ instead of $r(q,o)$ results in an unbiased gradient estimate. This is because $V(q)$ is constructed to be exactly the expected value of $Q(q,o)$, making it a valid baseline that preserves the expected gradient while reducing its variance.

An attentive reader might wonder why

$$
𝔼_{o\sim\pi_\theta(O \mid q)}\bigl[\nabla_\theta \log \pi_\theta(o \mid q)\bigr] = 0.
$$

Let's see why.

First, recall that for any probability distribution $\pi_\theta(o \mid q)$, we have:

$$
\sum_o \pi_\theta(o \mid q) = 1.
$$

Because this equality holds for all $\theta$, differentiating both sides with respect to $\theta$ gives:

$$
\nabla_\theta \left[\sum_o \pi_\theta(o \mid q)\right] = \sum_o \nabla_\theta \pi_\theta(o \mid q) = 0.
$$

Next, by applying the log-derivative trick, we know that:

$$
\nabla_\theta \pi_\theta(o \mid q) = \pi_\theta(o \mid q) \nabla_\theta \log \pi_\theta(o \mid q).
$$

Substituting this into the previous sum yields:

$$
\sum_o \pi_\theta(o \mid q) \nabla_\theta \log \pi_\theta(o \mid q) = 0.
$$

Recognizing that this weighted sum is exactly the expectation of the score function under $\pi_\theta$, we have:

$$
𝔼_{o\sim\pi_\theta(O \mid q)}\bigl[\nabla_\theta \log \pi_\theta(o \mid q)\bigr] = 0.
$$

This derivation is often referred to as the **score function property** and is a key step in the **likelihood ratio trick**, which is widely used in methods like policy gradient.

---

In this refined formulation of the REINFORCE algorithm, outputs that yield rewards higher than the expected baseline (i.e., a positive advantage) are reinforced, while those yielding lower-than-expected rewards (i.e., a negative advantage) are discouraged. This adjustment improves the assignment of credit to the generated outputs and reduces the variance in the gradient estimates, leading to a more stable learning process.

This formulation, which leverages the difference between $Q(q,o)$ and $V(q)$ to compute the advantage, is a fundamental component of [actor-critic methods](actor-critic.md). In actor-critic approaches, the policy $\pi_\theta$ (the actor) selects outputs, while a separate mechanism (the critic) estimates $V(q)$. The critic’s evaluation provides the necessary information to calculate the advantage, thereby guiding the policy updates more effectively.
