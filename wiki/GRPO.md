Below is the full revised tutorial with the additional detailed explanation integrated. In the new sections, we include a simple numerical example illustrating the computations for a concrete action and state, and we explain how, in text generation, many actions and states are combined to form the overall KL divergence penalty. No existing explanations are reduced or removed.



# Building the GRPO Algorithm from First Principles

Mathematical reasoning is a challenging task for language models because it requires both accurate quantitative reasoning and careful multi-step problem solving. In our work, we aim to boost the mathematical capabilities of a model by combining standard supervised fine-tuning with reinforcement learning (RL). In particular, our Group Relative Policy Optimization (GRPO) algorithm is designed to maximize reward signals (which encourage correct reasoning) while regularizing the model so that it does not deviate too abruptly from a trusted reference behavior. GRPO is a variant of Proximal Policy Optimization (PPO) that foregoes a separate critic model in favor of estimating baselines from groups of outputs sampled per prompt—reducing memory requirements while leveraging the comparative nature of reward signals.

In the following sections, we introduce the two main ingredients of GRPO—the policy gradient loss and the Kullback–Leibler (KL) divergence penalty—and show how they combine to form our per-token loss. We then describe how model outputs are converted to log probabilities, how completions are generated and masked (so that tokens after an end-of-sequence marker are ignored), and finally how all these pieces are integrated into a full training loop that includes reinforcement learning.



## 1. Policy Gradient Loss

In reinforcement learning, our goal is to adjust the model’s parameters so that its outputs receive higher rewards. In the simplest policy gradient method, when the model produces an action (or token) with log probability 
$$
\log \pi(a|s)
$$
in state $s$, and we observe a reward $R$, the parameter update is proportional to

$$
\nabla \log \pi(a|s) \, R.
$$

Since raw rewards may have varying scales, it is common practice to compute a normalized advantage $A$ that indicates how much better an action is compared to the average. For example, if the reward for a completion is $R$ and the average reward is $\bar{R}$ with standard deviation $\sigma_R$, one common normalization is

$$
A = \frac{R - \bar{R}}{\sigma_R + \epsilon}.
$$

In our GRPO setting, the policy gradient loss for each token is weighted by the advantage. This means that tokens leading to better-than-average completions are reinforced, while tokens that contribute to poorer outputs are suppressed.

### Motivation

In the context of mathematical reasoning, the correctness of each step is crucial. By normalizing rewards into advantages, the model can focus on reinforcing tokens that contribute to solving a problem step by step. Moreover, if a particular completion is especially strong compared to other samples for the same prompt, its tokens receive more reinforcement.



## 2. Kullback–Leibler (KL) Divergence Penalty

While pushing the model toward higher rewards is important, we must also ensure that the model’s behavior does not deviate too radically from a stable reference (e.g., the model’s state after supervised fine-tuning). To control this, GRPO incorporates a KL divergence penalty that measures the difference between the current policy $\pi_{\text{policy}}$ and a reference policy $\pi_{\text{ref}}$:

$$
\mathrm{KL}\bigl(\pi_{\text{policy}}(\cdot|s) \,\|\, \pi_{\text{ref}}(\cdot|s)\bigr) = \sum_{a} \pi_{\text{policy}}(a|s) \log \frac{\pi_{\text{policy}}(a|s)}{\pi_{\text{ref}}(a|s)}.
$$

This equation considers all possible actions (or tokens) $a$ available at a given state $s$. The term $\pi_{\text{policy}}(a|s)$ weights each action by its probability under the current policy, while the logarithmic term $\log \frac{\pi_{\text{policy}}(a|s)}{\pi_{\text{ref}}(a|s)}$ quantifies how much more (or less) likely the action is under the current policy compared to the reference. When the two distributions are identical for all actions, the KL divergence is zero, which means that no penalty is imposed. However, if the current policy deviates from the reference, the KL divergence increases and penalizes the update accordingly.

### Detailed Derivation of the KL Divergence Penalty

To understand precisely how each component of the KL divergence is obtained and calculated, we begin with the standard definition. Assume that at a given state $s$, the current policy defines a probability distribution over the possible actions (or tokens) $a$ as $\pi_{\text{policy}}(a|s)$, and similarly, the reference policy defines $\pi_{\text{ref}}(a|s)$. The KL divergence is defined as

$$
\mathrm{KL}(P \,\|\, Q) = \sum_{a} P(a) \log \frac{P(a)}{Q(a)},
$$

where in our case, $P(a) = \pi_{\text{policy}}(a|s)$ and $Q(a) = \pi_{\text{ref}}(a|s)$. This expression tells us the average extra “cost” (in terms of log probability) incurred when using the reference distribution to encode data actually generated by the current policy.

For many reinforcement learning applications, especially in our GRPO formulation, we are interested in the penalty imposed on the token that was actually generated. For that token, we define

$$
\Delta = \log \pi_{\text{policy}}(a|s) - \log \pi_{\text{ref}}(a|s).
$$

Here, $a$ is the concrete action (token) taken in state $s$. In a simple numerical example, suppose for a given token $a$ at state $s$ the current policy assigns a probability of 0.3 and the reference policy assigns a probability of 0.1. Then:

- Compute the log probabilities:
  - $\log(0.3) \approx -1.20397$
  - $\log(0.1) \approx -2.30259$
- Thus, 
  $$
  \Delta \approx -1.20397 - (-2.30259) = 1.09862.
  $$

This positive value indicates that the current policy assigns a higher probability to the token $a$ than the reference policy does.

Next, we exponentiate this difference to obtain the ratio:

$$
r = \exp(\Delta) = \exp(1.09862) \approx 3.
$$

This means that the current policy's probability is about three times larger than that of the reference for this token. In other words,

$$
r = \frac{\pi_{\text{policy}}(a|s)}{\pi_{\text{ref}}(a|s)} \approx 3.
$$

This numerical example clarifies that if the current policy favors a token significantly more than the reference policy, the computed ratio $r$ will be greater than 1.

### Combining Many Actions and States in Text Generation

In text generation, a model generates a sequence of tokens. At each time step $t$, the state $s_t$ consists of the history of previously generated tokens $ (a_1, a_2, \dots, a_{t-1}) $, and the action $a_t$ is the token generated at that time step. For every token in the sequence, we compute a corresponding $\Delta_t$ and $r_t$ as described above. These per-token values are then aggregated (e.g., summed or averaged) over the entire sequence to yield a measure of how the full generated sequence deviates from the reference. This aggregation is used in the overall loss to penalize sequences that deviate too far from the baseline behavior, ensuring that the GRPO regularization applies not only at a single time step but across the entire text.



## 3. The Per-Token Loss Function

By combining the policy gradient loss and the KL divergence penalty, we obtain the following per-token loss:

$$
L_{\text{token}} = -\Bigl( \exp(\Delta) \cdot A - \beta \cdot \text{KL} \Bigr),
$$

where 
$$
\Delta = \log \pi_{\text{policy}}(a|s) - \log \pi_{\text{ref}}(a|s).
$$
In this expression:
- $A$ is the normalized advantage.
- $\exp(\Delta)$ rescales the gradient to emphasize tokens that are more likely under the current policy relative to the reference.
- The KL divergence is computed (in a per-token form) as

$$
\text{KL} = \exp(\Delta) - \Delta - 1,
$$

a formulation that is both computationally convenient and unbiased.

### Motivation and Intuition

This loss function achieves a careful balance: the first term rewards tokens that increase the probability of higher-reward completions, while the second term penalizes those tokens if they stray too far from the reference model. In other words, the model is “pushed” toward correct reasoning but is kept in check by the regularization term.



## 4. Converting Model Outputs to Log Probabilities

Language models output raw scores (logits) for each token in the vocabulary. To transform these into stable probabilities (and subsequently log probabilities), we use the log‑softmax function. Here is a Python function that performs this conversion and then gathers the log probability corresponding to a specific token:

```python
import torch
import torch.nn.functional as F

def selective_log_softmax(logits, input_ids):
    # Compute log probabilities along the vocabulary axis.
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log probability for the token specified in input_ids.
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    # Remove the extra dimension.
    return selected_log_probs.squeeze(-1)
```

### Motivation

Working in log space not only improves numerical stability but also simplifies gradient computations. It is especially helpful when integrating the KL penalty directly into the loss function.



## 5. Generating and Masking Completions

When generating text, the model is given a prompt and then produces a completion. We need to align the logits with the generated tokens and also create a mask so that we only compute the loss on the relevant tokens (i.e., stopping at the first end-of-sequence marker). The following code snippet shows how to generate multiple completions per prompt and apply the mask:

```python
def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    device = next(model.parameters()).device
    # Tokenize prompts with left padding.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prompt_length = prompt_ids.size(1)
    # Repeat prompts to generate multiple completions per prompt.
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    # Generate completions (prompt + generated tokens).
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Extract completion tokens by removing the prompt portion.
    completion_ids = outputs[:, prompt_length:]
    # Create a mask so that tokens after the first EOS are ignored.
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def create_completion_mask(completion_ids, eos_token_id):
    # Identify positions where EOS token appears.
    is_eos = completion_ids == eos_token_id
    # For each sequence, find the first occurrence of EOS.
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    # Create a mask: 1 for tokens up to and including the first EOS, 0 thereafter.
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    return completion_mask
```

### Motivation

By generating multiple completions per prompt, we can robustly estimate rewards and compute a group-relative advantage. The mask ensures that only the tokens before (and including) the first end-of-sequence token contribute to the loss, which is particularly important in tasks where outputs vary in length.



## 6. Defining the GRPO Loss

The core idea of GRPO is to use the group of completions (generated for each prompt) to compute a relative baseline for rewards. Rather than training a separate value function (as in PPO), we use the average reward of the completions as a baseline. This is well motivated in mathematical reasoning, where outputs are naturally compared against each other.

Given:
$$
\Delta = \log \pi_{\text{policy}}(a|s) - \log \pi_{\text{ref}}(a|s),
$$
a per-token KL divergence computed as 
$$
\text{KL} = \exp(\Delta) - \Delta - 1,
$$
and a normalized advantage $A$ computed from the rewards of a group of outputs, the per-token loss becomes

$$
L_{\text{token}} = -\Bigl( \exp(\Delta) \cdot A - \beta \cdot \text{KL} \Bigr).
$$

When we sample a group of $G$ completions for each prompt, the final loss is averaged over both tokens and completions.

### Motivation

GRPO leverages the fact that, for a given question, multiple completions allow us to form a relative ranking. By comparing each sample to the group average, we can derive an advantage that better reflects how much each token contributes to a high-quality solution. This “group relative” approach is especially aligned with human judgment, where answers are typically judged in comparison to one another.



## 7. Training with GRPO: The Full Loop

Finally, we integrate all components into a training loop. During each training iteration, a batch of prompts is sampled; for each prompt, multiple completions are generated; log probabilities for each token are computed with both the current model and the reference model; a reward function (which might be based on correctness, step-by-step evaluation, or even human preference) assigns a reward to each generated output; the advantages are normalized across the group; the per-token loss is computed and averaged; and then the model is updated using backpropagation. Finally, the reference model is updated to follow the current model, ensuring that the KL penalty remains effective.

Below is the complete training function:

```python
import copy
import torch
import random

def grpo_loss(model, ref_model, tokenizer, batch_samples, reward_function,
              beta=0.1, num_generations=4, max_completion_length=32):
    device = next(model.parameters()).device
    # Extract prompts from batch samples.
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    # Generate completions and masks.
    prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
        model, tokenizer, prompts, num_generations, max_completion_length
    )
    # Concatenate prompt and completion tokens.
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    # Compute log probabilities using the reference model (no gradient).
    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    # Compute log probabilities using the current model.
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    
    # Decode completions for reward evaluation.
    formatted_completions = [
        [{'content': tokenizer.decode(ids, skip_special_tokens=True)}]
        for ids in completion_ids
    ]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1]
               for sample in batch_samples for _ in range(num_generations)]
    # Compute rewards for the generated completions.
    rewards = torch.tensor(
        reward_function(prompts=repeated_prompts, completions=formatted_completions, answer=answers),
        dtype=torch.float32,
        device=device
    )
    print("Average Reward:", rewards.mean().item())
    
    # Normalize rewards per prompt to obtain advantages.
    mean_rewards = rewards.view(-1, num_generations).mean(dim=1)
    std_rewards = rewards.view(-1, num_generations).std(dim=1)
    mean_rewards = mean_rewards.repeat_interleave(num_generations, dim=0)
    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
    
    # Compute per-token KL divergence:
    per_token_kl = torch.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1
    
    # Compute the policy gradient loss component.
    per_token_loss = torch.exp(token_log_probs - token_log_probs.detach()) * advantages.unsqueeze(1)
    
    # Combine both components into the final per-token loss.
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    
    # Average the loss per sequence using the completion mask.
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss

def train_with_grpo(model, tokenizer, train_data, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=128, beta=0.1,
                    learning_rate=5e-6):
    device = next(model.parameters()).device
    # Create a reference model with frozen parameters.
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(num_steps):
        # Sample a random batch from training data.
        batch_samples = random.sample(train_data, batch_size)
        # Compute the GRPO loss.
        loss = grpo_loss(
            model,
            ref_model,
            tokenizer,
            batch_samples,
            reward_function,  # Defined elsewhere.
            beta=beta,
            num_generations=num_generations,
            max_completion_length=max_completion_length
        )
        # Backpropagation and update.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        # Update the reference model to follow the current model.
        ref_model.load_state_dict(model.state_dict())
        if step % 5 == 0:
            print(f"Step {step}/{num_steps}, loss: {loss.item():.4f}")
        torch.cuda.empty_cache()
    return model
```

### Motivation

This training loop embodies several key ideas. Sampling multiple completions per prompt enables us to compute a group-based baseline (the average reward) and assign a relative advantage to each token. Periodically updating the reference model ensures that the KL penalty remains meaningful while keeping the computational overhead low. Additionally, using the "detachment trick" (i.e., `token_log_probs.detach()`) ensures that gradients flow only through the current model’s outputs rather than through the advantage computation.



## 8. Training with GRPO: A Balancing Act for Mathematical Reasoning

The GRPO algorithm improves the model’s outputs by maximizing rewards (thus pushing the model toward correct, high-quality mathematical reasoning) while regularizing its behavior through the KL penalty. By normalizing rewards into advantages across a group of completions, GRPO reinforces tokens that consistently contribute to strong performance and dampens those leading to poorer outcomes.

In contrast to standard PPO—which requires training a separate value (critic) model—GRPO leverages group comparisons directly from the sampled completions. This design not only reduces memory usage but also aligns more naturally with the comparative nature of many reward models (which often judge two completions relative to one another).



## 9. Summary and Future Directions

To summarize, the GRPO algorithm combines two main components to boost mathematical reasoning: the policy gradient loss, which reinforces tokens that lead to higher-than-average rewards, and the KL divergence penalty, which regularizes the update so that the new policy remains close to a trusted reference. The group relative advantage, computed from multiple completions per prompt, provides a robust signal for each token. This entire framework is implemented in a reinforcement learning loop that is both efficient in memory usage and effective at improving mathematical reasoning.

The KL divergence term, given by

$$
\mathrm{KL}\bigl(\pi_{\text{policy}}(\cdot|s) \,\|\, \pi_{\text{ref}}(\cdot|s)\bigr) = \sum_{a} \pi_{\text{policy}}(a|s) \log \frac{\pi_{\text{policy}}(a|s)}{\pi_{\text{ref}}(a|s)},
$$

plays a critical role by measuring how much the current policy deviates from the reference. We detailed how this expression is obtained, breaking down the summation over all possible tokens, the weighting by the current policy, and the log-ratio that quantifies the discrepancy. In particular, for the token that was actually generated, we compute

$$
\Delta = \log \pi_{\text{policy}}(a|s) - \log \pi_{\text{ref}}(a|s),
$$

and then exponentiate it to obtain

$$
r = \exp(\Delta) = \frac{\pi_{\text{policy}}(a|s)}{\pi_{\text{ref}}(a|s)}.
$$

A simple numerical example illustrates this: if for a given token we have $\pi_{\text{policy}}(a|s)=0.3$ and $\pi_{\text{ref}}(a|s)=0.1$, then

$$
\Delta \approx \log(0.3) - \log(0.1) \approx 1.0986,
$$

and

$$
r \approx \exp(1.0986) \approx 3.
$$

This indicates that the current policy assigns three times the probability to the token compared to the reference. In text generation, each state $s_t$ is the history of tokens generated up to time $t$, and the action $a_t$ is the token produced at time $t$. We compute $\Delta_t$ and $r_t$ for each token and then combine these (by summing or averaging over the sequence) to form the overall KL penalty for the generated text. This ensures that although each $a$ and $s$ are computed at individual steps, they contribute to a unified evaluation of the entire sequence.

Future work may further refine data selection, reward modeling (for instance, using process supervision), and advanced sampling strategies to push the boundaries of mathematical reasoning even further.



By integrating the detailed derivation of the KL divergence penalty—including a numerical example and an explanation of how multiple tokens and states are combined—this tutorial now provides a comprehensive explanation of GRPO, covering every key component from foundational ideas to practical implementation.
