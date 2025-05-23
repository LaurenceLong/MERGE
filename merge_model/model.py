import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

from transformers import PreTrainedModel

from .configs import MERGEModelConfig
from .utils import (
    gumbel_sigmoid_st, multinomial_st, bottom_k_st,
    linear_anneal, calculate_mean_entropy, calculate_mean_margin,
    calculate_low_conf_ratio, calculate_self_ppl, get_round_embedding,
    bernoulli_kl_loss, calculate_mask_ratio
)
from .llama_components import LlamaDecoder, LlamaEncoder, GapEncoder # RoPE is now in LlamaAttention

logger = logging.getLogger(__name__)


class MERGELanguageModel(PreTrainedModel):
    config_class = MERGEModelConfig

    def __init__(self, config: MERGEModelConfig):
        super().__init__(config)
        self.config = config

        # 1. Word Embeddings (shared)
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # 2. Gate Module
        # Round embedding for Gate input
        self.round_embeddings = nn.Embedding(config.max_rounds_l + 1, config.round_emb_dim) # +1 for safety

        # Gate MLP: 2-layer
        # Actual input dim calculated in __post_init__ of config or here
        # gate_feature_dim = 1(H̄) + 1(margin) + 1(low_conf) + 1(ppl) + config.round_emb_dim
        # For now, use config.gate_feature_dim (ensure it's set correctly)
        _gate_input_dim = config.gate_feature_dim
        if _gate_input_dim <=0: # Fallback if not set
            _gate_input_dim = 4 + config.round_emb_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(_gate_input_dim, config.gate_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_mlp_hidden_dim, 1) # Outputs a single logit for g̃
        )

        # 3. Mask-Picker (M₁)
        # LLAMAdec(X_l) - a LlamaDecoder instance
        self.m1_llama_decoder = LlamaDecoder(config, num_layers_override=config.num_hidden_layers_decoder, causal=False) # Non-causal for scoring existing tokens
        # FFN for token-wise score s_i
        self.m1_score_ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.m1_ffn_intermediate_dim),
            nn.ReLU(),
            nn.Linear(config.m1_ffn_intermediate_dim, 1) # Outputs score s_i for each token
        )
        # Mask embedding for soft deletion: e[MASK]
        # self.m1_mask_embedding = self.word_embeddings.weight[config.mask_token_id].detach().clone()
        # This should be registered as a buffer if used like this, or taken directly in forward.

        # 4. Mask-Inserter (M₂)
        self.m2_gap_encoder = GapEncoder(config) # Takes S_hard_embeds
        self.m2_gap_logits_linear = nn.Linear(config.gap_encoder_hidden_dim, 1) # alpha_g from h_gap (per gap)
                                                                                # Output dim 1, then squeeze. Softmax over G gaps.

        # 5. Filling Network (E_MDLM)
        self.e_mdlm = LlamaEncoder(config, num_layers_override=config.num_hidden_layers_encoder)
        # Weight tying: E_MDLM's lm_head with word_embeddings
        if hasattr(self.e_mdlm, 'lm_head') and self.e_mdlm.lm_head is not None:
            self.e_mdlm.lm_head.weight = self.word_embeddings.weight
        else:
            logger.warning("E_MDLM (LlamaEncoder) does not have lm_head, cannot tie weights.")

        # 6. Loss functions
        # L_recon: Weighted CE. Will be custom calculated.
        # Other losses are calculated directly.
        self.ce_loss_fn_for_ppl = nn.CrossEntropyLoss(ignore_index=config.pad_token_id if config.pad_token_id is not None else -100)


        self.init_weights() # From PreTrainedModel

    def _init_weights(self, module):
        """ Initializes weights of modules. """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # RMSNorm weights are initialized to 1 by default in their class.

    def _calculate_gate_features(
        self,
        prev_e_mdlm_logits: Optional[torch.Tensor], # Logits from E_MDLM in previous round (B, L, V)
        prev_e_mdlm_attention_mask: Optional[torch.Tensor], # Attention mask for prev_e_mdlm_logits (B,L)
        current_X_l_tokens: Optional[torch.LongTensor], # Tokens for self_ppl reference if prev_logits are for them (B,L)
        round_idx: int, # Current round l
        batch_size: int
    ) -> torch.Tensor:
        device = self.word_embeddings.weight.device

        # Initialize features with zeros or default values
        mean_entropy_feat = torch.zeros(batch_size, 1, device=device)
        mean_margin_feat = torch.zeros(batch_size, 1, device=device)
        low_conf_ratio_feat = torch.zeros(batch_size, 1, device=device)
        self_ppl_feat = torch.zeros(batch_size, 1, device=device) # High PPL can be default (e.g. 1000)

        if prev_e_mdlm_logits is not None and current_X_l_tokens is not None:
            # Ensure prev_e_mdlm_attention_mask matches prev_e_mdlm_logits length
            # And current_X_l_tokens matches prev_e_mdlm_logits length for PPL calculation

            # H̄ (mean_entropy)
            mean_entropy_feat = calculate_mean_entropy(prev_e_mdlm_logits, prev_e_mdlm_attention_mask).view(batch_size, 1)
            # mean_margin(p1-p2)
            mean_margin_feat = calculate_mean_margin(prev_e_mdlm_logits, prev_e_mdlm_attention_mask).view(batch_size, 1)
            # low_conf_ratio(τ_H=1.5 nats)
            low_conf_ratio_feat = calculate_low_conf_ratio(prev_e_mdlm_logits, self.config.gate_low_conf_threshold_nats, prev_e_mdlm_attention_mask).view(batch_size, 1)
            # self_ppl (ppl_self)
            # This PPL is tricky: it's E_MDLM's prediction quality on the *previous* state X_{l-1}
            # So, prev_e_mdlm_logits should be predictions for current_X_l_tokens (which was X_{l-1})
            self_ppl_feat = calculate_self_ppl(prev_e_mdlm_logits, current_X_l_tokens, prev_e_mdlm_attention_mask, self.config.pad_token_id).view(batch_size, 1)

        # round_emb(l)
        round_l_tensor = torch.tensor([round_idx] * batch_size, device=device, dtype=torch.long)
        round_emb_feat = self.round_embeddings(round_l_tensor) # (B, round_emb_dim)

        # concat features
        # Ensure all features are (B, Dim)
        features = torch.cat([
            mean_entropy_feat,
            mean_margin_feat,
            low_conf_ratio_feat,
            self_ppl_feat,
            round_emb_feat
        ], dim=-1) # (B, gate_feature_dim)
        return features

    def _calculate_k_ins(self, length_S_hard: int,  max_len=20):
        return min(max(length_S_hard, 1), max_len)
    def _calculate_k_del(self, length_X_l: int) -> int:
        k = round(self.config.k_del_ratio * length_X_l)
        return int(torch.clamp(torch.tensor(k), self.config.k_del_min, self.config.k_del_max).item())

    def _calculate_k_ins(self, length_S_hard: int, length_Y_star: int) -> int:
        # calc_k(|S|)= clamp(round(0.25·(|Y*|−|S|)), 1, 20)
        if length_Y_star <= length_S_hard:
            return 0
        k = round(self.config.k_ins_target_ratio * (length_Y_star - length_S_hard))
        return int(torch.clamp(torch.tensor(k), self.config.k_ins_min, self.config.k_ins_max).item())

    def _insert_masks_at_gaps(
        self,
        S_hard_tokens: torch.LongTensor, # (B, L_s)
        S_hard_attention_mask: torch.LongTensor, # (B, L_s)
        gap_insertion_counts: torch.Tensor, # (B, G) where G = L_s + 1, counts of masks per gap
        max_output_len: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor]: # Returns (S_hat_hard_tokens, S_hat_hard_attention_mask)

        batch_size, L_s = S_hard_tokens.shape
        device = S_hard_tokens.device
        mask_token_id = self.config.mask_token_id
        pad_token_id = self.config.pad_token_id

        output_sequences = []
        for b_idx in range(batch_size):
            current_S_hard = S_hard_tokens[b_idx][S_hard_attention_mask[b_idx].bool()] # Valid tokens
            current_L_s = len(current_S_hard)
            current_gap_counts = gap_insertion_counts[b_idx] # (L_s + 1)

            temp_sequence = []
            # Gap 0 (before first token)
            num_masks_at_gap0 = int(current_gap_counts[0].item())
            temp_sequence.extend([mask_token_id] * num_masks_at_gap0)

            # Iterate through S_hard tokens and gaps after them
            for token_idx in range(current_L_s):
                temp_sequence.append(current_S_hard[token_idx].item())
                num_masks_at_gap = int(current_gap_counts[token_idx + 1].item())
                temp_sequence.extend([mask_token_id] * num_masks_at_gap)

            # Truncate or pad to max_output_len
            if len(temp_sequence) > max_output_len:
                final_sequence_tokens = temp_sequence[:max_output_len]
            else:
                final_sequence_tokens = temp_sequence + [pad_token_id] * (max_output_len - len(temp_sequence))

            output_sequences.append(torch.tensor(final_sequence_tokens, dtype=torch.long, device=device))

        S_hat_hard_tokens = torch.stack(output_sequences)
        S_hat_hard_attention_mask = (S_hat_hard_tokens != pad_token_id).long()
        return S_hat_hard_tokens, S_hat_hard_attention_mask

    def _fuse_sequences(
        self,
        S_hat_hard_tokens: torch.LongTensor, # (B, L_target) - sequence with hard masks
        x_soft_embeds: torch.Tensor, # (B, L_original, H) - soft embeddings from M1 (or original if no delete)
        # To map S_hat_hard to x_soft, we need to know which tokens in S_hat_hard are original vs new masks
        # This requires careful index management.
        # Let's assume S_hat_hard_tokens contains mask_token_id for newly inserted masks.
        # And x_soft_embeds corresponds to X_l (before M2's insertions).
        # The `fuse` rule: if token in S_hat_hard is new MASK -> use e[MASK]
        # else (it's an original token from S_hard) -> take from x_soft_embeds (this part is tricky)
        # x_soft_embeds was (B, L_Xl, H). S_hat_hard_tokens is (B, L_S_hat, H).
        # The `x_soft` from the scheme is `x̂_i = (1−p_del_i)·x_i + p_del_i·e[MASK]` which is aligned with X_l.
        # S_hard is a subsequence of X_l. S_hat_hard inserts masks into S_hard.
        # This implies `Ŝ_mixed` should be constructed carefully.
        # For now, a simplified fuse: if S_hat_hard has MASK, use MASK embedding. Otherwise, use S_hat_hard's own embedding.
        # This doesn't use x_soft_embeds as directly as the scheme implies.
        # A more faithful fuse would be:
        # 1. Identify original tokens in S_hat_hard (those not MASK from M2).
        # 2. For these, find their corresponding embeddings in x_soft_embeds (needs mapping from S_hard indices to X_l indices).
        # 3. For MASKs from M2, use e[MASK].
        # This is complex. Let's simplify for now:
        # If S_hat_hard[i] is MASK_TOKEN, mixed_embed[i] = e[MASK].
        # Else, mixed_embed[i] = embed(S_hat_hard[i]).
        # This means x_soft is not directly used here, which deviates from scheme.
        # Let's try to be more faithful:
        # `S_hat_hard` (B, L_final)
        # `x_soft` (B, L_Xl, H) - soft embeddings of original X_l tokens
        # `S_hard_indices_in_Xl` (B, L_S_hard) - indices mapping S_hard back to X_l
        # `S_hat_hard_is_new_mask` (B, L_final) - boolean mask indicating M2's new masks
    ) -> torch.Tensor: # Returns S_hat_mixed_embeds (B, L_target, H)

        # This is a placeholder for the complex fuse logic.
        # For now, just embed S_hat_hard_tokens.
        # The true fuse requires passing `S_hard_indices_in_Xl` and `S_hat_hard_is_new_mask_flags`
        # from the main forward pass.

        # Simplified version:
        S_hat_mixed_embeds = self.word_embeddings(S_hat_hard_tokens)
        mask_token_embed = self.word_embeddings.weight[self.config.mask_token_id]

        # Where S_hat_hard_tokens is MASK_ID, ensure it's the MASK embedding.
        # This is mostly for clarity as word_embeddings already does this.
        # The key is that non-MASK tokens in S_hat_hard should ideally come from x_soft.
        # This simplified version doesn't achieve the "soft token" aspect for non-new-MASKs.

        # TODO: Implement the full fuse logic as per Tip 4, using x_soft_embeds for non-newly-inserted MASKs.
        # This would require S_hat_hard to also carry information about which tokens are original vs. newly inserted.
        return S_hat_mixed_embeds


    def _get_weighted_ce_loss(
        self,
        logits: torch.Tensor, # (B, L, V) from E_MDLM
        Y_star_tokens: torch.LongTensor, # (B, L) ground truth
        S_hat_hard_tokens: torch.LongTensor, # (B, L) input to E_MDLM, contains MASKs
        attention_mask: torch.LongTensor, # (B, L) for Y_star / S_hat_hard
        current_mask_ratio: float # t_l for weighting
    ) -> torch.Tensor:
        # L_recon = weighted_CE_MDLM(logits, Y*, t_l) # Only MASK位
        # Identify MASK positions in S_hat_hard_tokens that should be predicted.
        mask_positions = (S_hat_hard_tokens == self.config.mask_token_id) & attention_mask.bool()

        if mask_positions.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Select logits and targets only at these mask positions
        masked_logits = logits[mask_positions] # (NumMasks, V)
        masked_targets = Y_star_tokens[mask_positions] # (NumMasks)

        if masked_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Calculate standard CE loss for these positions
        ce_loss_per_mask = F.cross_entropy(masked_logits, masked_targets, reduction='none')

        # Apply MDLM weighting: α_MDLM(t) = cos²(π·t/2)
        # Ensure t (mask_ratio) is between 0 and 1.
        t = max(0.0, min(1.0, current_mask_ratio))
        alpha_weight = (torch.cos(torch.tensor(math.pi * t / 2.0)) ** 2).item()

        weighted_loss = ce_loss_per_mask * alpha_weight
        return weighted_loss.mean()


    def forward(
        self,
        X_l_tokens: torch.LongTensor, # (B, L) current input sequence
        X_l_attention_mask: torch.LongTensor, # (B, L)
        prev_e_mdlm_output: Optional[Dict[str, torch.Tensor]], # Output from E_MDLM of prev round {logits, hidden_states}
                                                              # Or None for first round.
        round_idx: int,
        Y_star_tokens: torch.LongTensor, # (B, L) Ground truth for L_recon and teacher forcing
        # Annealing temperatures passed from training loop
        current_tau_gate: float,
        current_tau_del: float,
    ) -> Dict[str, Any]:

        batch_size, seq_len = X_l_tokens.shape
        device = X_l_tokens.device

        # --- 0. Initial Embeddings for X_l ---
        X_l_embeds = self.word_embeddings(X_l_tokens) # (B, L, H)

        # --- 1. Gate ---
        # For l=0, prev_e_mdlm_output is None. Gate features will be defaults.
        # Gate needs logits from previous E_MDLM output on X_{l-1} (which is current_X_l_tokens if l > 0).
        # And attention mask for those previous logits.
        prev_logits_for_gate = prev_e_mdlm_output["logits"] if prev_e_mdlm_output else None
        # If prev_e_mdlm_output exists, its hidden_states were for X_l_tokens (input to prev EMDLM)
        # and its attention_mask was X_l_attention_mask.
        # X_l_tokens here is X_l. prev_e_mdlm_output was for X_{l-1}.
        # This means we need X_{l-1} tokens to calculate PPL for Gate.
        # This is getting complicated. Let's assume Gate uses prev_e_mdlm_output["logits"] and current X_l_tokens
        # as X_{l-1} for PPL calculation.
        # The scheme says Gate(hidden_prev). hidden_prev is E_MDLM(Ŝ_mixed_{l-1}).
        # And features are mean_entropy(logits_{l-1}), etc.
        # So, prev_e_mdlm_output["logits"] and prev_e_mdlm_output["attention_mask"] (if it was for those logits)
        # And current_X_l_tokens are used as the reference for PPL of those logits.

        # For first round (round_idx=0), prev_e_mdlm_output is None.
        # Gate features will use defaults or simplified calculation.
        # The scheme says "首轮 g=0". We can enforce this or let the features lead to it.
        # Let's compute features. If prev_e_mdlm_output is None, features will be mostly zero/default.

        # To correctly calculate PPL for gate, `prev_e_mdlm_output['logits']` should be the prediction for `X_l_tokens`.
        # This means `X_l_tokens` here is effectively `X_{l-1}` if `prev_e_mdlm_output` is from processing `X_{l-1}`.
        # This implies `X_l_tokens` passed to `forward` is `X_true_current_round_input`.
        # And `prev_e_mdlm_output` is from `E_MDLM(X_true_current_round_input)` if it was processed.
        # This needs careful state passing in the training loop.

        # Simplified: Assume prev_e_mdlm_output["logits"] are predictions for some X_l_prime,
        # and X_l_tokens are those X_l_prime.
        # If round_idx == 0, prev_e_mdlm_output is None.
        # Gate features use (prev_logits, prev_mask_for_logits, tokens_for_prev_logits_ppl)
        # So, if prev_e_mdlm_output comes from X_prev, then X_l_tokens here should be X_prev for PPL.
        # This is confusing. Let's assume `X_l_tokens` is the input for *this* round.
        # `prev_e_mdlm_output` is the output of EMDLM from *last* round, whose input was `X_{l-1}`.
        # The Gate features are based on `logits_{l-1}`. So `prev_e_mdlm_output['logits']` are `logits_{l-1}`.
        # `self_ppl` for gate needs `(logits_{l-1}, X_{l-1})`.
        # We only have `X_l_tokens` (current input) and `Y_star_tokens` (overall target).
        # This implies the training loop needs to pass `X_{l-1}_tokens` if PPL is calculated this way.
        # For now, PPL for gate will be based on `prev_e_mdlm_output['logits']` and `X_l_tokens` (as a proxy for `X_{l-1}`).
        # This is a known simplification point.

        gate_features = self._calculate_gate_features(
            prev_e_mdlm_logits=prev_e_mdlm_output["logits"] if prev_e_mdlm_output else None,
            prev_e_mdlm_attention_mask=prev_e_mdlm_output["source_attention_mask"] if prev_e_mdlm_output else None, # EMDLM output should include the mask of its input
            current_X_l_tokens=X_l_tokens, # Used as X_{l-1} for PPL of prev_logits
            round_idx=round_idx,
            batch_size=batch_size
        )

        g_tilde_logit = self.gate_mlp(gate_features).squeeze(-1) # (B,)
        g_soft_prob = torch.sigmoid(g_tilde_logit) # (B,) for loss L_gate

        if round_idx == 0 and self.training: # Enforce g=0 (no edit) for the very first round during training as per scheme
            g_hard = torch.zeros_like(g_soft_prob)
        else:
            g_hard = gumbel_sigmoid_st(g_tilde_logit, current_tau_gate, hard=True) # (B,) 0 or 1

        # --- 2. Delete (M₁) ---
        # M1 operates on X_l_tokens and X_l_embeds
        # Output: S_hard_tokens, x_soft_embeds, p_del_i (for loss)
        # S_hard_indices_in_Xl (for fuse step)

        # Initialize outputs for this stage
        S_hard_tokens_m1 = X_l_tokens.clone()
        S_hard_attention_mask_m1 = X_l_attention_mask.clone()
        x_soft_embeds_m1 = X_l_embeds.clone() # Default: no change
        p_del_i_for_loss = torch.zeros_like(X_l_tokens, dtype=torch.float) # Default: no deletion prob
        # S_hard_indices_in_Xl_m1 = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) # Default: all original indices

        # Store original indices of X_l that are kept in S_hard
        # This is needed for the _fuse_sequences step later.
        # Let's make S_hard_tokens_m1 always max_seq_len, padded.
        # And S_hard_indices_in_Xl_m1 also max_seq_len, padded with -1.

        # Per-batch processing for g_hard, because K_del depends on individual sequence lengths
        temp_S_hard_list = []
        temp_S_hard_mask_list = []
        temp_x_soft_list = []
        temp_p_del_list = []
        # temp_S_hard_idx_map_list = [] # To map S_hard tokens back to X_l original positions

        for b_idx in range(batch_size):
            current_X_l_item_tokens = X_l_tokens[b_idx][X_l_attention_mask[b_idx].bool()]
            current_X_l_item_embeds = X_l_embeds[b_idx][X_l_attention_mask[b_idx].bool()]
            current_X_l_item_len = len(current_X_l_item_tokens)

            if current_X_l_item_len == 0: # Empty sequence
                temp_S_hard_list.append(torch.full((seq_len,), self.config.pad_token_id, dtype=torch.long, device=device))
                temp_S_hard_mask_list.append(torch.zeros((seq_len,), dtype=torch.long, device=device))
                temp_x_soft_list.append(torch.zeros((seq_len, self.config.hidden_size), dtype=torch.float, device=device))
                temp_p_del_list.append(torch.zeros((seq_len,), dtype=torch.float, device=device))
                # temp_S_hard_idx_map_list.append(torch.full((seq_len,), -1, dtype=torch.long, device=device))
                continue

            if g_hard[b_idx].item() == 1: # Edit Path for this item
                # 1. token-wise score s_i = FFN(LLAMAdec(X_l))[i]
                # LLAMAdec needs embeds and its own attention mask (causal + padding)
                # For M1, LLAMAdec is processing current_X_l_item_embeds
                m1_decoder_input_embeds = current_X_l_item_embeds.unsqueeze(0) # (1, L_item, H)
                # M1 decoder should be non-causal if scoring all tokens based on full context
                m1_dec_hidden_states = self.m1_llama_decoder(input_embeds=m1_decoder_input_embeds) # (1, L_item, H)

                s_i_logits = self.m1_score_ffn(m1_dec_hidden_states).squeeze(-1).squeeze(0) # (L_item,) scores (higher = keep)

                # 2. soft删除权重 p_del_i = σ(−s_i/τ_del)
                p_del_i = torch.sigmoid(-s_i_logits / current_tau_del) # (L_item,)
                p_del_i_for_loss_item = F.pad(p_del_i, (0, seq_len - current_X_l_item_len), value=0.0) # Pad for batch loss
                temp_p_del_list.append(p_del_i_for_loss_item)

                # 3. Soft token x̂_i = (1−p_del_i)·x_i + p_del_i·e[MASK]
                # This should be done at embedding level.
                e_mask = self.word_embeddings.weight[self.config.mask_token_id]
                x_soft_item_embeds = (1 - p_del_i.unsqueeze(-1)) * current_X_l_item_embeds + \
                                     p_del_i.unsqueeze(-1) * e_mask.unsqueeze(0) # (L_item, H)

                # 4. Hard skeleton (ST) - idx_del = BottomK_ST(s_i, K_del)
                # s_i are "keep" scores. So bottom K of s_i are to be deleted.
                K_del = self._calculate_k_del(current_X_l_item_len)
                if K_del > 0 and K_del < current_X_l_item_len : # Avoid deleting all or none if K_del is edge case
                    # bottom_k_st returns indices to be deleted (those with lowest scores)
                    # For ST, s_i needs gradient. It gets it from p_del_i used in x_soft_item_embeds.
                    # So, bottom_k_st can be hard bottom-k.
                    idx_to_delete = bottom_k_st(s_i_logits, K_del) # Returns indices within L_item

                    # Create S_hard by deleting these tokens
                    keep_mask = torch.ones(current_X_l_item_len, dtype=torch.bool, device=device)
                    keep_mask[idx_to_delete] = False
                    S_hard_item_tokens = current_X_l_item_tokens[keep_mask]
                    # S_hard_idx_map_item = torch.arange(current_X_l_item_len, device=device)[keep_mask]
                else: # No deletion or K_del is too large
                    S_hard_item_tokens = current_X_l_item_tokens
                    # S_hard_idx_map_item = torch.arange(current_X_l_item_len, device=device)

                # Pad S_hard_item_tokens and x_soft_item_embeds to seq_len for batching
                padded_S_hard_item_tokens = F.pad(S_hard_item_tokens, (0, seq_len - len(S_hard_item_tokens)), value=self.config.pad_token_id)
                padded_x_soft_item_embeds = F.pad(x_soft_item_embeds, (0, 0, 0, seq_len - current_X_l_item_len), value=0) # Pad H dim, then L dim
                # padded_S_hard_idx_map = F.pad(S_hard_idx_map_item, (0, seq_len - len(S_hard_idx_map_item)), value=-1)

                temp_S_hard_list.append(padded_S_hard_item_tokens)
                temp_S_hard_mask_list.append((padded_S_hard_item_tokens != self.config.pad_token_id).long())
                temp_x_soft_list.append(padded_x_soft_item_embeds)
                # temp_S_hard_idx_map_list.append(padded_S_hard_idx_map)

            else: # Generate Path (g_hard[b_idx] == 0, no deletion)
                S_hard_item_tokens = current_X_l_item_tokens
                x_soft_item_embeds = current_X_l_item_embeds # Original embeddings
                # S_hard_idx_map_item = torch.arange(current_X_l_item_len, device=device)

                padded_S_hard_item_tokens = F.pad(S_hard_item_tokens, (0, seq_len - len(S_hard_item_tokens)), value=self.config.pad_token_id)
                padded_x_soft_item_embeds = F.pad(x_soft_item_embeds, (0, 0, 0, seq_len - current_X_l_item_len), value=0)
                # padded_S_hard_idx_map = F.pad(S_hard_idx_map_item, (0, seq_len - len(S_hard_idx_map_item)), value=-1)

                temp_S_hard_list.append(padded_S_hard_item_tokens)
                temp_S_hard_mask_list.append((padded_S_hard_item_tokens != self.config.pad_token_id).long())
                temp_x_soft_list.append(padded_x_soft_item_embeds)
                temp_p_del_list.append(torch.zeros((seq_len,), dtype=torch.float, device=device)) # No deletion prob
                # temp_S_hard_idx_map_list.append(padded_S_hard_idx_map)

        S_hard_tokens_m1 = torch.stack(temp_S_hard_list)
        S_hard_attention_mask_m1 = torch.stack(temp_S_hard_mask_list)
        x_soft_embeds_m1 = torch.stack(temp_x_soft_list) # (B, seq_len, H) - this is x_soft for fuse
        p_del_i_for_loss = torch.stack(temp_p_del_list) # (B, seq_len)
        # S_hard_indices_in_Xl_m1 = torch.stack(temp_S_hard_idx_map_list) # (B, seq_len)

        # --- 3. Insert (M₂) ---
        # Operates on S_hard_tokens_m1. Output: S_hat_hard_tokens
        # K_ins = calc_k(|S_hard|, |Y*|)
        # For batching, K_ins needs to be calculated per item.
        # Then M2_insert. This also suggests per-batch item processing or careful batching.

        # Initialize for M2
        S_hat_hard_tokens = S_hard_tokens_m1.clone() # Default if K_ins is 0
        S_hat_hard_attention_mask = S_hard_attention_mask_m1.clone()
        z_g_for_loss_list = [] # List to store z_g for each batch item if K_ins > 0

        # Max length for S_hat_hard is original seq_len (or config.max_seq_len)
        # This is the target length for E_MDLM input.
        target_len_for_S_hat = seq_len

        temp_S_hat_hard_list = []
        temp_S_hat_hard_mask_list = []

        for b_idx in range(batch_size):
            current_S_hard_item_tokens = S_hard_tokens_m1[b_idx][S_hard_attention_mask_m1[b_idx].bool()]
            current_S_hard_item_len = len(current_S_hard_item_tokens)

            # Y_star_len for this item (non-padded length)
            current_Y_star_item_len = Y_star_tokens[b_idx][(Y_star_tokens[b_idx] != self.config.pad_token_id)].size(0)

            K_ins = self._calculate_k_ins(current_S_hard_item_len, current_Y_star_item_len)

            # Ensure K_ins doesn't make sequence exceed target_len_for_S_hat
            K_ins = min(K_ins, max(0, target_len_for_S_hat - current_S_hard_item_len))

            if K_ins > 0:
                # Embed S_hard for GapEncoder
                # Need unpadded S_hard_item_tokens for GapEncoder if it processes variable length
                # Or pass padded S_hard_tokens_m1[b_idx] with its mask
                current_S_hard_item_embeds = self.word_embeddings(current_S_hard_item_tokens.unsqueeze(0)) # (1, L_s_item, H)

                # h_gap = GapEncoder(S_hard_embeds) -> (1, L_s_item + 1, H_gap)
                h_gap = self.m2_gap_encoder(current_S_hard_item_embeds) # Pass actual S_hard embeds

                # alpha_g = Linear(h_gap) -> (1, L_s_item + 1, 1) then squeeze
                alpha_g_logits = self.m2_gap_logits_linear(h_gap).squeeze(-1) # (1, L_s_item + 1)

                # z_g = softmax(alpha_g)
                z_g_probs = F.softmax(alpha_g_logits, dim=-1) # (1, L_s_item + 1)
                z_g_for_loss_list.append(z_g_probs.squeeze(0)) # Store for loss (L_s_item+1,)

                # counts = Multinomial_ST(K_ins, z_g) -> (1, L_s_item+1)
                # Handle K_ins > max_masks_per_insertion_round_m2 (Tip 3) - simplified for now
                gap_insertion_counts = multinomial_st(z_g_probs, K_ins) # (1, L_s_item+1)

                # S_hat_hard_item_tokens, S_hat_hard_item_mask = self._insert_masks_at_gaps(...)
                # This helper needs S_hard_tokens (unpadded), its mask, counts, and target_len
                # Let's call it with the single batch item.
                # The helper _insert_masks_at_gaps needs to be adapted for single item or batch.
                # For now, let's assume we get S_hat_hard_item_tokens_unpadded.

                # Re-implementing _insert_masks_at_gaps logic here for single item:
                _temp_seq = []
                num_masks_gap0 = int(gap_insertion_counts[0, 0].item())
                _temp_seq.extend([self.config.mask_token_id] * num_masks_gap0)
                for token_idx in range(current_S_hard_item_len):
                    _temp_seq.append(current_S_hard_item_tokens[token_idx].item())
                    num_masks_gap_after = int(gap_insertion_counts[0, token_idx + 1].item())
                    _temp_seq.extend([self.config.mask_token_id] * num_masks_gap_after)

                # Pad/truncate S_hat_hard_item to target_len_for_S_hat
                if len(_temp_seq) > target_len_for_S_hat:
                    S_hat_hard_item_final_tokens = _temp_seq[:target_len_for_S_hat]
                else:
                    S_hat_hard_item_final_tokens = _temp_seq + \
                        [self.config.pad_token_id] * (target_len_for_S_hat - len(_temp_seq))

                S_hat_hard_item_tensor = torch.tensor(S_hat_hard_item_final_tokens, dtype=torch.long, device=device)
                temp_S_hat_hard_list.append(S_hat_hard_item_tensor)
                temp_S_hat_hard_mask_list.append((S_hat_hard_item_tensor != self.config.pad_token_id).long())

            else: # K_ins is 0
                # S_hat_hard is same as S_hard_m1 for this item
                # Pad S_hard_tokens_m1[b_idx] to target_len_for_S_hat
                unpadded_S_hard = S_hard_tokens_m1[b_idx][S_hard_attention_mask_m1[b_idx].bool()]
                padded_S_hard = F.pad(unpadded_S_hard, (0, target_len_for_S_hat - len(unpadded_S_hard)), value=self.config.pad_token_id)
                temp_S_hat_hard_list.append(padded_S_hard)
                temp_S_hat_hard_mask_list.append((padded_S_hard != self.config.pad_token_id).long())
                # Add placeholder for z_g if needed for consistent loss calculation structure
                # num_gaps = current_S_hard_item_len + 1
                # z_g_for_loss_list.append(torch.ones(num_gaps, device=device) / num_gaps) # Uniform if no insertion
                # Or handle None in loss calculation

        S_hat_hard_tokens = torch.stack(temp_S_hat_hard_list)
        S_hat_hard_attention_mask = torch.stack(temp_S_hat_hard_mask_list)

        # --- 4. Fuse & Fill (E_MDLM) ---
        # S_hat_mixed = fuse(S_hat_hard, x_soft_embeds_m1)
        # For now, simplified fuse: just embed S_hat_hard_tokens
        # x_soft_embeds_m1 is (B, seq_len, H), where seq_len is original X_l length.
        # S_hat_hard_tokens is (B, target_len_for_S_hat, H).
        # The fuse needs to map S_hat_hard tokens that were original in X_l to their x_soft_embeds.
        # This requires the S_hard_indices_in_Xl_m1 mapping.
        # This is a complex part. Using simplified fuse for now.
        S_hat_mixed_embeds = self._fuse_sequences(S_hat_hard_tokens, x_soft_embeds_m1) # Placeholder

        # current_mask_ratio for E_MDLM's pseudo-time condition
        # This should be mask ratio of S_hat_mixed_embeds's source tokens (S_hat_hard_tokens)
        current_mask_ratio_val = calculate_mask_ratio(S_hat_hard_tokens, self.config.mask_token_id, S_hat_hard_attention_mask)
        pseudo_time_cond_tensor = torch.tensor([current_mask_ratio_val] * batch_size, device=device).unsqueeze(-1) # (B,1)

        # E_MDLM forward pass
        e_mdlm_output = self.e_mdlm(
            input_embeds=S_hat_mixed_embeds,
            attention_mask=S_hat_hard_attention_mask, # Mask for S_hat_mixed
            pseudo_time_condition=pseudo_time_cond_tensor
        )
        logits_current_round = e_mdlm_output["logits"] # (B, L_target, V)
        hidden_states_current_round = e_mdlm_output["hidden_states"] # (B, L_target, H)

        # --- 5. Loss Calculation ---
        # a. L_recon (Weighted CE on MASK positions in S_hat_hard)
        # Y_star_tokens needs to be same length as S_hat_hard_tokens for direct comparison.
        # Assume Y_star_tokens is already padded/truncated to target_len_for_S_hat (which is seq_len).
        loss_recon = self._get_weighted_ce_loss(
            logits_current_round, Y_star_tokens, S_hat_hard_tokens, S_hat_hard_attention_mask, current_mask_ratio_val
        )

        # b. L_gate (Sparsity + Reward)
        # Reward signal for gate: relu(ΔH̄). ΔH̄ = H_target - H_current.
        # Or, if g_hard=1 (edit), reward is positive if entropy reduced or confidence increased.
        # This is still tricky. Let's use a simplified reward: e.g. if g_hard=1, reward is 1.
        # Or, reward = -mean_entropy(logits_current_round) if trying to minimize entropy.
        # Scheme: g_soft·relu(ΔH̄). Let ΔH̄ be a placeholder for now, or related to entropy change.
        # For simplicity, let's make reward = 1 if g_hard=1 and an edit was made (K_del > 0 or K_ins > 0).
        # This is a placeholder for a more meaningful reward.
        # A simple reward: if g_hard=1, reward is -(mean entropy of current logits)
        # This encourages edits that lead to more confident predictions.
        current_logits_mean_entropy = calculate_mean_entropy(logits_current_round, S_hat_hard_attention_mask)
        gate_reward_signal = -current_logits_mean_entropy # Lower entropy is better reward.

        loss_gate_sparsity = self.config.lambda_gate_sparsity * g_soft_prob.mean()
        # Only apply reward if g_soft_prob indicates a high chance of edit.
        loss_gate_reward = self.config.lambda_gate_reward * (g_soft_prob * F.relu(gate_reward_signal)).mean()
        loss_gate = loss_gate_sparsity + loss_gate_reward

        # c. L_comp (M₁ KL divergence for p_del_i vs target deletion ratio)
        # p_del_i_for_loss is (B, seq_len), X_l_attention_mask is (B, seq_len)
        # Target deletion ratio for M1
        loss_comp_m1 = self.config.lambda_m1_entropy * bernoulli_kl_loss(
            p_del_i_for_loss, self.config.m1_target_deletion_ratio, X_l_attention_mask
        )

        # d. L_ins (M₂ Entropy for z_g)
        if z_g_for_loss_list: # If any item had K_ins > 0
            mean_entropy_m2 = torch.stack([-torch.sum(zg * torch.log(zg + 1e-9), dim=-1) for zg in z_g_for_loss_list]).mean()
            loss_ins_m2 = self.config.lambda_m2_entropy * mean_entropy_m2
        else:
            loss_ins_m2 = torch.tensor(0.0, device=device)

        # Total Loss
        total_loss = loss_recon + loss_gate + loss_comp_m1 + loss_ins_m2

        return {
            "loss": total_loss,
            "loss_reconstruction": loss_recon.detach(),
            "loss_gate": loss_gate.detach(),
            "loss_m1_entropy": loss_comp_m1.detach(),
            "loss_m2_entropy": loss_ins_m2.detach(),
            "g_soft_prob_mean": g_soft_prob.mean().detach(),
            "g_hard_mean": g_hard.mean().detach(), # Proportion of batch items that chose edit path
            "logits": logits_current_round, # For next round's X_l if not teacher forcing
            "hidden_states": hidden_states_current_round, # For next round's Gate
            "S_hat_hard_tokens": S_hat_hard_tokens, # For next round's X_l
            "S_hat_hard_attention_mask": S_hat_hard_attention_mask, # For next round
            # Store the attention mask of the EMDLM input for gate PPL calculation in next round
            "source_attention_mask_for_emdlm_output": S_hat_hard_attention_mask.detach()
        }

    def greedy_fill(self, S_hat_hard_tokens: torch.LongTensor, logits: torch.Tensor) -> torch.LongTensor:
        """Fills MASK tokens in S_hat_hard_tokens using argmax of logits."""
        predictions = torch.argmax(logits, dim=-1) # (B, L)

        output_tokens = S_hat_hard_tokens.clone()
        mask_positions = (S_hat_hard_tokens == self.config.mask_token_id)
        output_tokens[mask_positions] = predictions[mask_positions]
        return output_tokens

    # --- Inference Methods (to be called by inference script) ---
    @torch.no_grad()
    def infer_gate(self, prev_e_mdlm_output: Optional[Dict[str, torch.Tensor]],
                   X_tokens_for_ppl: torch.LongTensor, X_attention_mask_for_ppl: torch.LongTensor,
                   round_idx: int, tau_edit: float) -> torch.Tensor: # Returns g (0 or 1) per batch item
        self.eval()
        batch_size = X_tokens_for_ppl.shape[0]
        gate_features = self._calculate_gate_features(
            prev_e_mdlm_logits=prev_e_mdlm_output["logits"] if prev_e_mdlm_output else None,
            prev_e_mdlm_attention_mask=prev_e_mdlm_output["source_attention_mask"] if prev_e_mdlm_output else None,
            current_X_l_tokens=X_tokens_for_ppl, # X_{l-1}
            round_idx=round_idx,
            batch_size=batch_size
        )
        g_tilde_logit = self.gate_mlp(gate_features).squeeze(-1)
        # Deterministic decision for inference: sigmoid > 0.5 (or use tau_edit if it's for threshold)
        # Scheme: τ_edit(l) = 0.4+0.1l. This might be a threshold for sigmoid(g_tilde_logit).
        # Or, if tau_edit is a temperature for Gumbel-Sigmoid, then it's different.
        # "g = Gate_infer(hidden_prev)" implies g is hard.
        # Let's use sigmoid > 0.5. tau_edit might be for a dynamic threshold.
        # For now, tau_edit is not used here, assuming it's for a different purpose or threshold.
        g = (torch.sigmoid(g_tilde_logit) > 0.5).float()
        return g

    @torch.no_grad()
    def infer_m1_delete_bottomk(self, X_l_tokens: torch.LongTensor, X_l_attention_mask: torch.LongTensor, K_del: int, tau_del_final: float) -> torch.LongTensor:
        # Returns S_hard_tokens (padded to original X_l_tokens.shape[1])
        self.eval()
        batch_size, seq_len = X_l_tokens.shape
        device = X_l_tokens.device

        S_hard_batch_list = []
        for b_idx in range(batch_size):
            current_X_l_item_tokens = X_l_tokens[b_idx][X_l_attention_mask[b_idx].bool()]
            current_X_l_item_len = len(current_X_l_item_tokens)

            if current_X_l_item_len == 0 or K_del == 0:
                S_hard_item_tokens = current_X_l_item_tokens
            else:
                current_X_l_item_embeds = self.word_embeddings(current_X_l_item_tokens.unsqueeze(0))
                m1_dec_hidden_states = self.m1_llama_decoder(input_embeds=current_X_l_item_embeds)
                s_i_logits = self.m1_score_ffn(m1_dec_hidden_states).squeeze(-1).squeeze(0) # (L_item,)

                actual_K_del = min(K_del, current_X_l_item_len -1) # Ensure at least one token remains
                if actual_K_del > 0:
                    idx_to_delete = torch.topk(s_i_logits, actual_K_del, largest=False, sorted=False).indices
                    keep_mask = torch.ones(current_X_l_item_len, dtype=torch.bool, device=device)
                    keep_mask[idx_to_delete] = False
                    S_hard_item_tokens = current_X_l_item_tokens[keep_mask]
                else:
                    S_hard_item_tokens = current_X_l_item_tokens

            padded_S_hard_item = F.pad(S_hard_item_tokens, (0, seq_len - len(S_hard_item_tokens)), value=self.config.pad_token_id)
            S_hard_batch_list.append(padded_S_hard_item)

        return torch.stack(S_hard_batch_list)


    @torch.no_grad()
    def infer_m2_insert_masks(self, S_tokens: torch.LongTensor, S_attention_mask: torch.LongTensor, K_ins: int) -> torch.Tensor:
        # Returns counts_g (B, G) for inserting masks. Does not do the insertion.
        self.eval()
        batch_size, _ = S_tokens.shape

        all_gap_counts = []
        for b_idx in range(batch_size):
            current_S_item_tokens = S_tokens[b_idx][S_attention_mask[b_idx].bool()]
            current_S_item_len = len(current_S_item_tokens)
            num_gaps = current_S_item_len + 1

            if K_ins > 0 and current_S_item_len < self.config.max_seq_len : # Only if K_ins > 0 and space available
                current_S_item_embeds = self.word_embeddings(current_S_item_tokens.unsqueeze(0))
                h_gap = self.m2_gap_encoder(current_S_item_embeds)
                alpha_g_logits = self.m2_gap_logits_linear(h_gap).squeeze(-1) # (1, num_gaps)
                z_g_probs = F.softmax(alpha_g_logits, dim=-1)

                # For inference, usually pick top-K gaps or sample. Scheme implies Multinomial_ST.
                # Here, let's use argmax for simplicity if K_ins=1, or distribute based on probs.
                # A simple way: sample K_ins times from z_g_probs.
                gap_insertion_counts = torch.zeros_like(z_g_probs, dtype=torch.float) # (1, num_gaps)
                if K_ins > 0 and z_g_probs.sum() > 0: # Ensure probs are valid
                    # Sample K_ins indices with replacement
                    sampled_indices = torch.multinomial(z_g_probs.squeeze(0), K_ins, replacement=True) # (K_ins)
                    for idx in sampled_indices:
                        gap_insertion_counts[0, idx] += 1
                all_gap_counts.append(gap_insertion_counts.squeeze(0)) # (num_gaps)
            else:
                all_gap_counts.append(torch.zeros(num_gaps, device=S_tokens.device))

        # Pad all_gap_counts to max_num_gaps in batch if lengths differ
        # Max possible gaps = max_seq_len + 1
        max_gaps_in_batch = S_tokens.shape[1] + 1
        padded_gap_counts = []
        for gc in all_gap_counts:
            padded_gc = F.pad(gc, (0, max_gaps_in_batch - len(gc)), value=0)
            padded_gap_counts.append(padded_gc)

        return torch.stack(padded_gap_counts) # (B, max_gaps_in_batch)

    @torch.no_grad()
    def greedy_fill_with_confidence(self, S_hat_hard_tokens: torch.LongTensor, logits: torch.Tensor) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Fills MASK tokens and returns filled tokens and mean confidence of filled tokens."""
        self.eval()
        predictions = torch.argmax(logits, dim=-1) # (B, L)
        probs = torch.softmax(logits, dim=-1) # (B, L, V)

        output_tokens = S_hat_hard_tokens.clone()
        mask_positions = (S_hat_hard_tokens == self.config.mask_token_id)

        output_tokens[mask_positions] = predictions[mask_positions]

        # Calculate confidence: prob of the predicted token at MASK positions
        confidences_at_masks = probs[mask_positions] # (NumMasks_in_batch, V)
        if confidences_at_masks.numel() > 0:
            # Get prob of the chosen token (argmax)
            chosen_token_probs = confidences_at_masks[torch.arange(confidences_at_masks.size(0)), predictions[mask_positions]]
            mean_confidence = chosen_token_probs.mean()
        else:
            mean_confidence = torch.tensor(1.0, device=logits.device) # No masks, perfect confidence

        return output_tokens, mean_confidence

