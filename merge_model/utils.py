from typing import Optional

import torch
import torch.nn.functional as F
import math

def gumbel_noise(
        shape: torch.Size,
        device: torch.device,
        eps: float = 1e-9
) -> torch.Tensor:
    """
    Generates Gumbel noise Gumbel(0,1).
    """
    uniform_samples = torch.rand(shape, device=device)
    return -torch.log(-torch.log(uniform_samples + eps) + eps)

def gumbel_sigmoid_st(logits: torch.Tensor, temperature: float, hard: bool = True, gumbel_noise_val: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Gumbel-Sigmoid Straight-Through Estimator.
    Input: Logits (before sigmoid).
    Output: Sampled binary tensor (0 or 1).
    """
    if gumbel_noise_val is None:
        gumbel_noise_val = gumbel_noise(logits.shape, logits.device)
    
    y_soft = torch.sigmoid((logits + gumbel_noise_val) / temperature)

    if hard:
        # Straight-through
        y_hard = (y_soft > 0.5).float()
        # STE: y_hard in forward, y_soft in backward
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft

def multinomial_st(probs: torch.Tensor, num_samples: int, hard: bool = True) -> torch.Tensor:
    """
    Straight-Through Multinomial sampling.
    probs: (B, N) or (N) - probabilities for N categories.
    num_samples: K - number of items to sample.
    Returns: (B, N) or (N) - counts of samples for each category.
             The sum of counts will be num_samples.
    This is complex for ST. A simpler version for ST might be sampling K indices with replacement.
    The scheme implies 'hard count (ST)'.
    If we need to sample K items and get their counts for G gaps:
    This can be seen as K independent categorical samples, then count.
    Or, one multinomial sample of K items.
    For ST, often gumbel_softmax is used for selection, then made hard.
    Let's use torch.multinomial and try to make it ST if possible, or just use its output.
    A true ST for multinomial counts is non-trivial.
    Alternative: K Gumbel-Softmax selections if we are selecting K *distinct* slots.
    If it's "distribute K items into G bins", then it's one Multinomial(K, probs_G).
    """
    if num_samples == 0:
        return torch.zeros_like(probs, dtype=torch.float)

    # For ST, we'd typically use Gumbel-Top-K or similar for selection.
    # For Multinomial counts, this is harder.
    # Let's assume for now this is a differentiable relaxation if possible,
    # or that the gradient path is handled by the overall loss.
    # The scheme implies `Multinomial_ST(K_ins, z_g)`.
    # `torch.multinomial` itself is not differentiable wrt probs for `replacement=True` in a way that ST helps easily.
    
    # Simplified approach: Sample K indices, then use ST on selection.
    # This is more like K independent choices if we use Gumbel-Max K times.
    # The scheme's `counts = Multinomial_ST(K_ins, z_g)` implies `counts` is a vector of length G.

    # A common ST approach for sampling K items is to use K Gumbel-Softmaxes (if items are distinct)
    # or use a continuous relaxation of the multinomial distribution.
    # For now, let's use the direct torch.multinomial and assume the gradient path for z_g
    # is primarily through the entropy regularization L_ins.
    # If a hard ST version is needed, it would involve something like:
    # 1. logits = log(probs + eps)
    # 2. perturbed_logits = (logits + gumbel_noise_like(logits)) / temp
    # 3. soft_counts_or_samples = some_softmax_based_relaxation_of_multinomial(perturbed_logits, K)
    # 4. hard_counts = discretize(soft_counts_or_samples)
    # 5. return hard_counts - soft_counts_or_samples.detach() + soft_counts_or_samples

    # Using torch.multinomial directly (non-ST for now, relies on L_ins for z_g grads)
    # This samples indices. We need counts.
    if probs.dim() == 1:
        probs = probs.unsqueeze(0) # Batch dim
    
    batch_size, num_categories = probs.shape
    counts = torch.zeros_like(probs, dtype=torch.float)

    for i in range(batch_size):
        if probs[i].sum() == 0: # Avoid error if all probs are zero
            # Distribute K_ins uniformly if possible, or assign to first if K_ins=1
            if num_samples > 0 and num_categories > 0:
                 counts[i, :num_samples % num_categories] = 1 # Simplified fallback
            continue
        
        # Ensure probs sum to 1 for multinomial
        current_probs = probs[i] / (probs[i].sum() + 1e-9)
        
        # Sample K_ins indices with replacement
        # `torch.multinomial` samples indices, not counts directly for a fixed K_ins total.
        # To get counts for K_ins items, we can do K_ins independent categorical samples.
        if num_samples > 0 :
            sampled_indices = torch.multinomial(current_probs, num_samples, replacement=True) # Shape: (num_samples)
            for idx in sampled_indices:
                counts[i, idx] += 1
    
    if probs.dim() == 1: # If original was 1D
        return counts.squeeze(0)
    return counts # This is hard, but gradient won't flow through the sampling process itself.

def bottom_k_st(scores: torch.Tensor, k: int, hard: bool = True) -> torch.Tensor:
    """
    Selects indices of bottom K scores using Straight-Through.
    scores: (B, L) or (L)
    k: number of items to select (smallest scores)
    Returns: A boolean mask (B, L) or (L) where True indicates a selected (bottom K) item.
             Or, returns indices. Scheme implies `idx_del = BottomK_ST(s_i, K_del)`.
    To make it ST, we need a soft version.
    s_i are scores where lower means more likely to delete.
    So we want top-K of these scores for deletion.
    Or, if s_i are "importance" scores, then bottom-K of importance.
    Scheme: `p_del_i = σ(−s_i/τ_del)`. `s_i` from `FFN(LLAMAdec(X_l))`.
    If `s_i` is higher, `p_del_i` is lower. So `s_i` is like a "keep_score".
    Then `BottomK_ST(s_i, K_del)` means delete tokens with lowest "keep_scores".
    """
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    # Invert scores to find "top-K smallest" as "top-K largest of negative scores"
    # This is for selection, not for gradient.
    # For ST, usually we have logits for selection.
    # Let `selection_logits = -scores`. We want to pick top K of these.
    
    # Soft selection: use a differentiable top-k approximation (e.g., from ST-Gumbel-softmax literature)
    # For simplicity and following common ST patterns:
    # 1. Get hard indices
    # 2. Create a soft differentiable version (e.g. by scaling scores with temperature)
    # 3. Apply STE
    
    # Hard selection of indices
    _, bottom_k_indices = torch.topk(scores, k, dim=-1, largest=False, sorted=False)
    
    # This function is expected to return indices for deletion.
    # The ST part would apply if we were creating a soft mask that then gets hardened.
    # `idx_del = BottomK_ST(s_i, K_del)`
    # `S_hard = X_l.delete(idx_del)`
    # The gradient for `s_i` needs to come from `L_recon` via `x̂_i`.
    # The `idx_del` itself doesn't need to be ST if `x̂_i` handles the gradient path for `s_i`.
    # However, if `BottomK_ST` is meant to produce differentiable *selection probabilities*
    # which are then used, it's different.
    # Given `p_del_i = σ(−s_i/τ_del)` and `x̂_i = (1−p_del_i)·x_i + p_del_i·e[MASK]`,
    # the gradient for `s_i` already flows through `p_del_i`.
    # The `idx_del` is for the hard branch.
    # So, `BottomK_ST` might just be hard bottom-k, and ST is a misnomer here,
    # or it implies that the *process* of choosing K should be ST.
    # If we need ST selection, we'd use something like Gumbel-TopK.

    # For now, returning hard indices, assuming ST is handled by the soft branch `x̂_i`.
    # If ST is strictly for `idx_del` to make the choice of K differentiable:
    #   - This is complex. Often involves relaxing the TopK operation.
    #   - E.g., using a soft sort and selecting. Or a regularized optimal transport approach.
    # Given the context of `x̂_soft` providing a gradient path, `BottomK_ST` might
    # just mean "hard bottom K selection, where the scores `s_i` have gradients from elsewhere".
    # Let's return indices directly.
    return bottom_k_indices


def linear_anneal(
        current_step: int,
        start_value: float,
        end_value: float,
        total_annealing_steps: int,
        warmup_steps: int = 0, # Steps before annealing starts
        start_anneal_step: int = 0 # Offset for when annealing phase begins
) -> float:
    if total_annealing_steps <= 0:
        return end_value
    
    # Adjust current_step relative to the actual start of annealing
    step = max(0, current_step - start_anneal_step)

    if step < warmup_steps: # If current_step is within its own warmup phase for this parameter
        return start_value

    effective_step = step - warmup_steps
    effective_total_steps = total_annealing_steps - warmup_steps

    if effective_total_steps <= 0:
        return end_value

    progress_ratio = min(effective_step / effective_total_steps, 1.0)
    return start_value + progress_ratio * (end_value - start_value)

# --- Gate Input Feature Functions ---
def calculate_mean_entropy(logits: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculates mean token entropy from logits. H(p) = -sum(p log p)"""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_entropy = -torch.sum(probs * log_probs, dim=-1) # (B, L)
    
    if attention_mask is not None:
        masked_entropy = token_entropy * attention_mask
        mean_entropy_val = masked_entropy.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1) # (B,)
    else:
        mean_entropy_val = token_entropy.mean(dim=-1) # (B,)
    return mean_entropy_val.mean() # Scalar over batch

def calculate_mean_margin(logits: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculates mean (p1 - p2) where p1 is prob of top1, p2 is prob of top2."""
    probs = torch.softmax(logits, dim=-1) # (B, L, V)
    top2_probs, _ = torch.topk(probs, 2, dim=-1) # (B, L, 2)
    
    margins = top2_probs[..., 0] - top2_probs[..., 1] # (B, L)
    
    if attention_mask is not None:
        masked_margins = margins * attention_mask
        mean_margin_val = masked_margins.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1) # (B,)
    else:
        mean_margin_val = margins.mean(dim=-1) # (B,)
    return mean_margin_val.mean() # Scalar over batch

def calculate_low_conf_ratio(logits: torch.Tensor, threshold_nats: float, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculates ratio of tokens with entropy > threshold_nats."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_entropy = -torch.sum(probs * log_probs, dim=-1) # (B, L)
    
    low_conf_tokens = (token_entropy > threshold_nats).float() # (B, L)
    
    if attention_mask is not None:
        num_low_conf = (low_conf_tokens * attention_mask).sum(dim=-1) # (B,)
        num_valid_tokens = attention_mask.sum(dim=-1).clamp(min=1) # (B,)
        ratio = num_low_conf / num_valid_tokens
    else:
        num_low_conf = low_conf_tokens.sum(dim=-1) # (B,)
        num_valid_tokens = torch.tensor(logits.shape[1], device=logits.device).float()
        ratio = num_low_conf / num_valid_tokens
    return ratio.mean() # Scalar over batch

def calculate_self_ppl(logits: torch.Tensor, targets: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, pad_token_id: int = -100) -> torch.Tensor:
    """Calculates perplexity of the model on its own generated/current targets."""
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), 
                           ignore_index=pad_token_id, reduction='none')
    loss = loss.view(targets.shape) # (B, L)

    if attention_mask is not None:
        masked_loss = loss * attention_mask
        avg_loss = masked_loss.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1) # (B,)
    else:
        # Assuming no padding if no mask, or that pad_token_id handles it
        non_pad_mask = (targets != pad_token_id).float()
        masked_loss = loss * non_pad_mask
        avg_loss = masked_loss.sum(dim=-1) / non_pad_mask.sum(dim=-1).clamp(min=1)

    ppl = torch.exp(avg_loss)
    return ppl.mean() # Scalar over batch

def get_round_embedding(round_l: int, embedding_dim: int, max_rounds: int = 10, device: torch.device = 'cpu') -> torch.Tensor:
    """Simple sinusoidal or learnable embedding for round number."""
    # Using a simplified fixed sinusoidal embedding for now
    # For learnable, use nn.Embedding(max_rounds, embedding_dim)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)).to(device)
    pe = torch.zeros(embedding_dim, device=device)
    pe[0::2] = torch.sin(round_l * div_term)
    pe[1::2] = torch.cos(round_l * div_term)
    return pe

def bernoulli_kl_loss(p_pred: torch.Tensor, p_target: float, attention_mask: Optional[torch.Tensor] = None, eps: float = 1e-9) -> torch.Tensor:
    """
    KL divergence D_KL(P_target || P_pred) for Bernoulli distributions.
    p_pred: predicted probabilities (e.g., p_del_i).
    p_target: target probability (scalar).
    Returns: KL divergence per element, possibly masked.
    KL(p||q) = p log(p/q) + (1-p) log((1-p)/(1-q))
    Here, p is target, q is pred.
    """
    term1 = p_target * (torch.log(p_target + eps) - torch.log(p_pred + eps))
    term2 = (1 - p_target) * (torch.log(1 - p_target + eps) - torch.log(1 - p_pred + eps))
    kl_div = term1 + term2
    
    if attention_mask is not None:
        kl_div = kl_div * attention_mask / attention_mask.sum().clamp(min=1)
        return kl_div.sum() # Sum over sequence and batch, then average by num valid tokens
    else:
        return kl_div.mean() # Average over all elements

def calculate_mask_ratio(token_ids: torch.Tensor, mask_token_id: int, attention_mask: Optional[torch.Tensor] = None) -> float:
    """Calculates the ratio of MASK tokens in the sequence."""
    is_mask = (token_ids == mask_token_id).float()
    if attention_mask is not None:
        num_masks = (is_mask * attention_mask).sum()
        num_tokens = attention_mask.sum().clamp(min=1)
    else:
        num_masks = is_mask.sum()
        num_tokens = token_ids.numel()
    
    return (num_masks / num_tokens).item() if num_tokens > 0 else 0.0

