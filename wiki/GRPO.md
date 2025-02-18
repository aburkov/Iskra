# Building the GRPO Algorithm from First Principles

In reinforcement learning, our goal is to adjust a model (or **policy**) so that it produces outputs that maximize a reward signal. However, while pursuing higher rewards, we also want to ensure that the model’s behavior does not diverge too far from what has been proven to work. GRPO (Group Relative Policy Optimization) achieves this balance by combining two main ingredients. The first is the policy gradient loss, which encourages the model to generate outputs that lead to higher rewards. The second is the Kullback-Leibler (KL) divergence penalty, which keeps the model’s behavior close to that of a stable reference model, thereby preventing overly drastic changes.

In the following sections, we introduce each of these concepts and describe how they come together to form the GRPO loss.

## Policy Gradient Loss

In reinforcement learning, the policy gradient method adjusts the model’s parameters by estimating the gradient of the expected reward. Suppose the model generates a token with log probability $\log \pi(a|s)$ (where $a$ is the action or token and $s$ is the state or context). When we receive a reward $R$, the update rule is proportional to

$$
\nabla \log \pi(a|s) \, R.
$$

Rather than using raw rewards, it is common to work with a normalized signal called the advantage, denoted $A$. The advantage quantifies how much better an action is compared to the average. If we have $R$ as the reward for a completion and $\bar{R}$ as the average reward, then the advantage can be defined as

$$
A = \frac{R - \bar{R}}{\sigma_R + \epsilon},
$$

where $\sigma_R$ is the standard deviation of the rewards (to normalize the scale) and $\epsilon$ is a small constant to avoid division by zero. In our GRPO setting, the policy gradient loss for each token is weighted by $A$. This means that tokens leading to completions that are better than average (that is, with positive advantage) are reinforced, while tokens contributing to worse completions are suppressed.

## Kullback-Leibler (KL) Divergence Penalty

While the policy gradient loss helps push the model toward higher rewards, it might lead to drastic changes in behavior. To control this, GRPO uses a KL divergence penalty between the current policy and a stable reference model. The KL divergence between two distributions $P$ (current policy) and $Q$ (reference) is defined as

$$
\text{KL}(P \,\|\, Q) = \sum_{a} P(a) \log \frac{P(a)}{Q(a)}.
$$

In our context, the reference model serves as a snapshot of a previous state of the model, ensuring that the current model does not deviate too far. The penalty is weighted by a parameter $\beta$, which controls the strength of this regularization.

## The Per-Token Loss

By combining the two ingredients—the policy gradient component and the KL penalty—we arrive at the following per-token loss:

$$
L_{\text{token}} = -\left( \exp(\Delta) \cdot A - \beta \cdot \text{KL} \right),
$$

where $\Delta$ is the difference in log probabilities between the current model and the reference model. In other words, 

$$
\Delta = \log \pi_{\text{policy}}(a|s) - \log \pi_{\text{ref}}(a|s).
$$

Here, $A$ is the normalized advantage and $\beta$ is the weight of the KL penalty. The KL divergence is computed per token as

$$
\text{KL} = \exp(\Delta) - \Delta - 1.
$$

This form ensures that the model is rewarded (via the policy gradient term) for increasing the probability of tokens that contribute to high rewards and that it is penalized for straying too far from the behavior of the reference model.

## Converting Model Outputs to Log Probabilities

Language models produce raw scores, called logits, for each token in the vocabulary. To turn these into probabilities (and then work in log space), we apply the log-softmax function. This process improves numerical stability and simplifies gradient computations. The following Python function performs this conversion and selects the log probability for the generated token:

```python
import torch
import torch.nn.functional as F

def selective_log_softmax(logits, input_ids):
    # Compute log probabilities along the vocabulary axis
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log probability for the specified token in input_ids
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    # Remove the extra dimension
    return selected_log_probs.squeeze(-1)
```

## Computing Log Probabilities for Completion Tokens

When generating text, the model receives a prompt and generates additional text (the completion). We need the log probabilities for these generated tokens. The following function calls the model, aligns the logits with the generated tokens, and applies the selective log-softmax:

```python
def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    # Get logits from the model with one extra token for alignment
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    # Discard the extra logit
    logits = logits[:, :-1, :]
    # Isolate the tokens corresponding to the completion
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    # Return the selective log probabilities
    return selective_log_softmax(logits, input_ids)
```

## Creating a Mask for the Completion

In many cases, the model stops generating text once an end-of-sequence (`EOS`) token appears. To compute the loss only on the relevant tokens, a mask is created that marks tokens after the first `EOS` as irrelevant. The following function produces such a mask. It identifies the positions where the `EOS` token appears and then builds a binary mask so that only the tokens up to and including the first `EOS` are considered in the loss computation:

```python
def create_completion_mask(completion_ids, eos_token_id):
    # Identify positions where EOS token appears
    is_eos = completion_ids == eos_token_id
    # Initialize tensor with sequence length
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    # Update indices where EOS is found
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    # Create sequence indices for each token
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    # Build the mask: 1 for tokens up to the first EOS, 0 otherwise
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    return completion_mask
```

## Generating Multiple Completions

To robustly estimate the reward and the resulting advantage, the algorithm generates multiple completions for each prompt. This diversity allows for better reward evaluation. The function below tokenizes a prompt, repeats it to produce several completions, and then generates the text. It first tokenizes the prompts with left padding, then repeats the prompt tokens and the corresponding attention masks so that multiple completions per prompt are generated. Finally, it extracts the completion tokens by removing the prompt portion and applies a mask to ignore tokens after the first `EOS`:

```python
def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    device = next(model.parameters()).device
    # Tokenize prompts with left padding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prompt_length = prompt_ids.size(1)
    # Repeat prompts to generate multiple completions per prompt
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    # Generate completions (prompt + generated tokens)
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Extract completion tokens by removing the prompt portion
    completion_ids = outputs[:, prompt_length:]
    # Apply mask to ignore tokens after the first EOS
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask
```

## Defining the GRPO Loss

The next step is to combine all the above pieces into the final GRPO loss. The loss function uses the reference model to compute stable log probabilities, the policy model for current probabilities, and a reward function that evaluates the quality of the generated completions. The overall GRPO loss is computed as follows.

First, prompts are extracted from the batch samples and completions along with their masks are generated. The prompt tokens and completion tokens are then concatenated, and the corresponding attention masks are combined. The log probabilities for the completion tokens are computed both using the reference model (without gradient tracking) and the current model.

Completions are decoded for reward evaluation, and rewards are computed by the reward function. These rewards are then normalized to obtain advantages. The per-token KL divergence is computed by using the formula $\text{KL} = \exp(\Delta) - \Delta - 1$, where $\Delta$ is the difference between the log probabilities from the reference model and the current model.

Next, the policy gradient loss component is computed with a trick to ensure gradients only flow through the current model's log probabilities.

Finally, both components are combined into the final per-token loss, and the loss is averaged per sequence using the completion mask. The complete function is given below:

```python
import copy

def grpo_loss(model, ref_model, tokenizer, batch_samples, reward_function,
              beta=0.1, num_generations=4, max_completion_length=32):
    device = next(model.parameters()).device
    # Extract prompts from batch samples
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    # Generate completions and masks
    prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
        model, tokenizer, prompts, num_generations, max_completion_length
    )
    # Concatenate prompt and completion tokens
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    # Compute log probabilities using the reference model (no gradient)
    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    # Compute log probabilities using the current model
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    
    # Decode completions for reward evaluation
    formatted_completions = [
        [{'content': tokenizer.decode(ids, skip_special_tokens=True)}]
        for ids in completion_ids
    ]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1]
               for sample in batch_samples for _ in range(num_generations)]
    # Compute rewards for the generated completions
    rewards = torch.tensor(
        reward_function(prompts=repeated_prompts, completions=formatted_completions, answer=answers),
        dtype=torch.float32,
        device=device
    )
    print("Average Reward:", rewards.mean().item())
    
    # Normalize rewards per prompt to obtain advantages
    mean_rewards = rewards.view(-1, num_generations).mean(dim=1)
    std_rewards = rewards.view(-1, num_generations).std(dim=1)
    mean_rewards = mean_rewards.repeat_interleave(num_generations, dim=0)
    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)
    
    # Compute per-token KL divergence:
    # KL = exp(Δ) - Δ - 1, where Δ = ref_token_log_probs - token_log_probs
    per_token_kl = torch.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1
    
    # Compute the policy gradient loss component.
    # Note: We use a trick to ensure gradients only flow through token_log_probs:
    per_token_loss = torch.exp(token_log_probs - token_log_probs.detach()) * advantages.unsqueeze(1)
    
    # Combine both components into the final per-token loss:
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    
    # Average the loss per sequence using the completion mask (ignoring tokens after EOS)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss
```

Let us break down some key parts in a narrative form. The difference in log probabilities, denoted by $\Delta$, is computed as

$$
\Delta = \log \pi_{\text{ref}}(a|s) - \log \pi_{\text{policy}}(a|s).
$$

Then, the KL divergence per token is calculated using the formula

$$
\text{KL} = \exp(\Delta) - \Delta - 1.
$$

Finally, the overall loss per token is defined as

$$
L_{\text{token}} = -\left( \exp(\Delta) \cdot A - \beta \cdot \text{KL} \right).
$$

## Training the Model with GRPO

The final step is to integrate everything into a training loop. A deep copy of the model is used as the reference model. During each training step, a batch of prompts is sampled, completions are generated, and the GRPO loss is computed. The model is updated using backpropagation, and then the reference model is updated to follow the current model. The training loop is implemented as follows.

First, the reference model is created by deep copying the current model, and its parameters are frozen. An optimizer is then set up for the current model.

In each iteration of the training loop, a random batch is sampled from the training data, the GRPO loss is computed using the functions defined above, and then backpropagation is performed with gradient clipping.

Finally, the reference model is updated with the state of the current model, and progress is printed every few steps. The complete training function is given below:

```python
def train_with_grpo(model, tokenizer, train_data, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=128, beta=0.1,
                    learning_rate=5e-6):
    device = next(model.parameters()).device
    # Create a reference model with frozen parameters
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(num_steps):
        # Sample a random batch from training data
        batch_samples = random.sample(train_data, batch_size)
        # Compute the GRPO loss
        loss = grpo_loss(
            model,
            ref_model,
            tokenizer,
            batch_samples,
            reward_function,  # Defined elsewhere
            beta=beta,
            num_generations=num_generations,
            max_completion_length=max_completion_length
        )
        # Backpropagation and update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        # Update the reference model
        ref_model.load_state_dict(model.state_dict())
        if step % 5 == 0:
            print(f"Step {step}/{num_steps}, loss: {loss.item():.4f}")
        torch.cuda.empty_cache()
    return model
```

GRPO algorithm improves the model's outputs by maximizing rewards, regularizing behavior, and leveraging normalized advantages. 

The policy gradient loss pushes the model to generate tokens that lead to higher rewards, while the KL divergence penalty ensures that the updated model remains close to a stable reference model.

By normalizing rewards into advantages, the algorithm emphasizes tokens that make a positive difference. This careful balancing act—using both the policy gradient and the KL penalty—is at the heart of GRPO, ensuring both progress and stability in model training.
