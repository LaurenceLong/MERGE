import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# --- Rotary Positional Embeddings (RoPE) ---
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The 'position_ids' argument is needed if using static cache optimization in HF transformers
    # For dynamic calculation, it's often just based on seq_len
    cos = cos[position_ids].unsqueeze(1) # [seq_len, dim] -> [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1) # [seq_len, dim] -> [bs, 1, seq_len, dim]
    
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)

    # Correct application for q, k shapes like (bs, num_heads, seq_len, head_dim)
    q_embed = (q * cos[:, :, -q.shape[-2]:, :]) + (rotate_half(q) * sin[:, :, -q.shape[-2]:, :])
    k_embed = (k * cos[:, :, -k.shape[-2]:, :]) + (rotate_half(k) * sin[:, :, -k.shape[-2]:, :])
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None): # Added config
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.max_position_embeddings = config.max_seq_len # RoPE needs this
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx # For potential future use with kv caching

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=self.max_position_embeddings, 
            base=self.rope_theta
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None, # (B, 1, L_q, L_k)
            position_ids: Optional[torch.LongTensor] = None, # (B, L_q)
            encoder_hidden_states: Optional[torch.Tensor] = None, # For cross-attention
            encoder_attention_mask: Optional[torch.Tensor] = None, # (B, 1, L_q, L_kv)
            # past_key_value: Optional[Tuple[torch.Tensor]] = None, # For kv caching
            # output_attentions: bool = False,
            # use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if encoder_hidden_states is not None:
            # Cross-attention
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(encoder_hidden_states).view(bsz, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(encoder_hidden_states).view(bsz, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
            # RoPE is typically not applied in cross-attention query/key
        else:
            # Self-attention
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
            
            cos, sin = self.rotary_emb(value_states, seq_len=q_len) # value_states is just for device/dtype
            if position_ids is None:
                position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # K/V Caching logic would go here if use_cache=True
        # For this implementation, we assume no caching for simplicity in MERGE context for now.
        # if past_key_value is not None:
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # present_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            # HF attention mask is (bsz, 1, q_len, kv_seq_len)
            # Values are 0 for tokens to attend to, -inf for tokens to ignore
            attn_weights = attn_weights + attention_mask
            # attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)) # prevent nan

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # return attn_output, attn_weights if output_attentions else None, present_key_value
        return attn_output # Simplified output for now


class LlamaMLP(nn.Module):
    def __init__(self, config): # Added config
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size # Usually 4 * hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu # SwiGLU

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # ... other params for attention like past_key_value, output_attentions, use_cache
        encoder_hidden_states: Optional[torch.Tensor] = None, # For cross-attention
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: # Simplified output
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)

        attn_output = self.self_attn(
            hidden_states_norm,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        # attn_output, self_attn_weights, present_key_value = self.self_attn(...)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states_norm = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states_norm)
        hidden_states = residual + mlp_output

        return hidden_states #, (self_attn_weights, present_key_value) if output_all_attentions else hidden_states


class LlamaStack(nn.Module): # Generic stack of LlamaLayers
    def __init__(self, config, num_layers_override: int = None, is_decoder: bool = False):
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder # For causal mask generation if needed by default
        num_layers = num_layers_override if num_layers_override is not None else config.num_hidden_layers_default
        self.layers = nn.ModuleList([LlamaLayer(config, layer_idx=i) for i in range(num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor, # input_embeds
        attention_mask: Optional[torch.Tensor] = None, # This is the padding mask (B, L)
        position_ids: Optional[torch.LongTensor] = None,
        # ... other params
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None, # This is (B, 1, L_q, L_kv)
    ) -> torch.Tensor:
        
        input_shape = hidden_states.size()[:-1]
        seq_length = input_shape[-1]

        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        # Prepare attention mask for LlamaAttention (it expects (B, 1, L_q, L_k))
        # If it's a decoder and causal, this needs to be combined with causal mask.
        # The original LlamaDecoder in the prompt handled causal mask itself.
        # For a generic stack, the caller should provide the correct mask.
        # If `attention_mask` is (B,L), expand it.
        if attention_mask is not None and attention_mask.dim() == 2:
            # expand mask to [bsz, 1, q_len, kv_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype, tgt_len=seq_length)

        # If this stack is a decoder and needs causal masking by default:
        if self.is_decoder:
            causal_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
            if attention_mask is not None:
                attention_mask = attention_mask + causal_mask # Additive masks
            else:
                attention_mask = causal_mask
        
        for layer_module in self.layers:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask # Pass this along
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

# Helper functions for mask manipulation (from HuggingFace Transformers)
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaEncoder(nn.Module):
    """ E_MDLM: 12-L Transformer Encoder + MLM Head + 伪时间条件 """
    def __init__(self, config, num_layers_override: Optional[int] = None):
        super().__init__()
        self.config = config
        num_layers = num_layers_override if num_layers_override is not None else config.num_hidden_layers_encoder
        
        self.stack = LlamaStack(config, num_layers_override=num_layers, is_decoder=False) # Encoder is not causal by default
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Pseudo-time condition embedding
        if config.pseudo_time_emb_dim > 0:
            # Simple approach: a small MLP to project scalar mask_ratio to an embedding
            # Or, nn.Embedding if mask_ratio is discretized.
            # Using a linear layer for continuous mask_ratio -> embedding
            self.pseudo_time_projector = nn.Linear(1, config.pseudo_time_emb_dim)
            self.pseudo_time_combiner = nn.Linear(config.hidden_size + config.pseudo_time_emb_dim, config.hidden_size)
        else:
            self.pseudo_time_projector = None
            self.pseudo_time_combiner = None


    def forward(self, input_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, pseudo_time_condition: Optional[torch.Tensor] = None) -> dict:
        """
        input_embeds: (B, L, H)
        attention_mask: (B, L) padding mask, 1 for valid, 0 for pad. Will be converted.
        pseudo_time_condition: (B, 1) or scalar, representing current_mask_ratio (t_l)
        """
        
        # Add pseudo-time condition to embeddings
        if self.pseudo_time_projector and pseudo_time_condition is not None:
            if pseudo_time_condition.ndim == 0: # scalar
                pseudo_time_condition = pseudo_time_condition.unsqueeze(0).unsqueeze(0).expand(input_embeds.size(0), 1) # B, 1
            elif pseudo_time_condition.ndim == 1: # B
                pseudo_time_condition = pseudo_time_condition.unsqueeze(1) # B, 1
            
            time_embed = self.pseudo_time_projector(pseudo_time_condition.float()) # (B, pseudo_time_emb_dim)
            time_embed_expanded = time_embed.unsqueeze(1).expand(-1, input_embeds.size(1), -1) # (B, L, pseudo_time_emb_dim)
            
            combined_embeds = torch.cat([input_embeds, time_embed_expanded], dim=-1)
            input_embeds = self.pseudo_time_combiner(combined_embeds) # Project back to hidden_size

        # LlamaStack expects attention_mask in (B, 1, L_q, L_k) format where 0 is attend, -inf is ignore
        # The input `attention_mask` is typically (B,L) where 1 is attend, 0 is pad.
        if attention_mask is not None and attention_mask.dim() == 2:
            # This will convert (B,L) with 1=attend,0=pad to (B,1,L,L) with 0=attend, -inf=ignore
            expanded_attention_mask = _expand_mask(attention_mask, input_embeds.dtype, tgt_len=input_embeds.size(1))
        else:
            expanded_attention_mask = attention_mask # If already in correct format or None

        hidden_states = self.stack(
            hidden_states=input_embeds, 
            attention_mask=expanded_attention_mask
        )
        logits = self.lm_head(hidden_states)
        return {"hidden_states": hidden_states, "logits": logits}

    def get_input_embeddings(self): # Not used if forward takes embeds
        return None 
    def set_input_embeddings(self, value): # Not used
        pass


class LlamaDecoder(nn.Module):
    """ LLAMAdec(X_l) for M1 Mask-Picker """
    def __init__(self, config, num_layers_override: Optional[int] = None, causal: bool = True):
        super().__init__()
        self.config = config
        num_layers = num_layers_override if num_layers_override is not None else config.num_hidden_layers_decoder
        # The scheme implies M1's LLAMAdec processes X_l. It might be self-attention.
        # If it's autoregressive over X_l to decide for each token, then causal=True.
        # "token-wise score s_i = FFN(LLAMAdec(X_l))[i]"
        # This sounds like each output hidden state s_i depends on X_l[:i] if causal.
        # Or, if non-causal, each s_i depends on all X_l.
        # Let's assume non-causal for M1's score prediction for now, as it's picking from existing.
        # If it needs to be causal (e.g. like a traditional decoder predicting next), then is_decoder=True.
        # The original code had `causal=True` for mask_picker_decoder. Let's stick to that.
        self.stack = LlamaStack(config, num_layers_override=num_layers, is_decoder=causal)

    def forward(
        self,
        input_embeds: torch.Tensor, # X_l embeddings
        attention_mask: Optional[torch.Tensor] = None, # (B,L) padding mask for X_l
        # encoder_hidden_states etc. if it were a cross-attending decoder
    ) -> torch.Tensor: # Returns hidden_states (B, L, H)

        if attention_mask is not None and attention_mask.dim() == 2:
            expanded_attention_mask = _expand_mask(attention_mask, input_embeds.dtype, tgt_len=input_embeds.size(1))
            if self.stack.is_decoder: # Add causal mask if it's a causal decoder stack
                causal_mask = _make_causal_mask(input_embeds.shape[:-1], input_embeds.dtype, device=input_embeds.device)
                expanded_attention_mask = expanded_attention_mask + causal_mask
        elif self.stack.is_decoder: # No padding mask, but still need causal
             expanded_attention_mask = _make_causal_mask(input_embeds.shape[:-1], input_embeds.dtype, device=input_embeds.device)
        else:
            expanded_attention_mask = None

        output_hidden_states = self.stack(
            hidden_states=input_embeds,
            attention_mask=expanded_attention_mask
        )
        return output_hidden_states


# --- Gap Encoder for M2 ---
class GapEncoder(nn.Module):
    """ Encodes S_hard to get G = |S_hard|+1 gap representations """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.gap_representation_dim = config.gap_encoder_hidden_dim

        # Learnable embeddings for gaps before the first token and after the last token
        self.bos_gap_emb = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.eos_gap_emb = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        # A small transformer to process S_hard with these boundary gap embeddings
        # Or simpler: use S_hard's own embeddings + projections.
        # Scheme: h_gap = GapEncoder(S_hard).
        # Let's try a simple FFN per token of S_hard, plus the BOS/EOS gaps.
        # This means h_gap_i is derived from S_hard_i.
        # h_gap will have |S_hard| elements from S_hard, +2 for BOS/EOS. Total |S_hard|+2.
        # The scheme says G = L-K+1, which is |S_hard|+1.
        # This implies one of the BOS/EOS gaps is implicit or handled differently.
        
        # Let's make it |S_hard|+1 gap representations.
        # Gap 0: before S_hard[0]
        # Gap i: after S_hard[i-1] (and before S_hard[i]) for i=1..|S_hard|
        
        # Simpler: Use S_hard's hidden states.
        # h_S_hard (B, L_s, H). We need (B, L_s+1, H_gap).
        # h_gap_0 = FFN_bos(context_vector or learnable_bos_gap_vector)
        # h_gap_i = FFN_internal(h_S_hard_i) for i=0..L_s-1 (representing gap *after* S_hard_i)
        # This gives L_s internal gaps + 1 BOS gap = L_s+1 gaps.

        self.gap_bos_projector = nn.Linear(self.hidden_size, self.gap_representation_dim) # Takes a context, e.g. mean S_hard
        self.gap_internal_projector = nn.Linear(self.hidden_size, self.gap_representation_dim) # Takes h_S_hard_i

        # Alternative: A small RNN/Transformer over S_hard_embeds
        # For now, using projections from S_hard_embeds itself.
        # This GapEncoder will be called with S_hard_embeds.

    def forward(self, S_hard_embeds: torch.Tensor, S_hard_attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        S_hard_embeds: (B, L_s, H)
        S_hard_attention_mask: (B, L_s)
        Returns: h_gap (B, L_s + 1, H_gap)
        """
        batch_size, L_s, H = S_hard_embeds.shape

        # Gap before first token: use a learnable embedding or projection of global context
        # For simplicity, let's use a learnable BOS gap embedding expanded to batch size
        # This bos_gap_emb is (1,1,H). We need (B,1,H_gap)
        # Let's assume self.bos_gap_emb is already H_gap.
        # bos_gap = self.bos_gap_emb.expand(batch_size, 1, -1) # (B, 1, H_gap)
        
        # A better BOS gap: project the first token's embedding (if exists) or a global S_hard context
        if L_s > 0:
            # Use first valid token embedding to inform BOS gap
            # Or mean pool S_hard_embeds if mask provided
            if S_hard_attention_mask is not None:
                masked_S_hard_embeds = S_hard_embeds * S_hard_attention_mask.unsqueeze(-1)
                global_S_hard_context = masked_S_hard_embeds.sum(dim=1) / S_hard_attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
            else:
                global_S_hard_context = S_hard_embeds.mean(dim=1) # (B, H)
            
            projected_bos_context = self.gap_bos_projector(global_S_hard_context) # (B, H_gap)
            bos_gaps = projected_bos_context.unsqueeze(1) # (B, 1, H_gap)
        else: # S_hard is empty
            # Fallback: use a generic learnable BOS gap if S_hard is empty
            # This requires a learnable parameter for empty S_hard context
            # For now, if L_s is 0, h_gap will be just one BOS gap.
            # Let's assume S_hard is not empty for typical M2 usage.
            # If L_s=0, then L_s+1 = 1. We need one gap representation.
            # This could be a single learnable vector.
             dummy_context = torch.zeros(batch_size, H, device=S_hard_embeds.device) # Or a learnable param
             projected_bos_context = self.gap_bos_projector(dummy_context)
             bos_gaps = projected_bos_context.unsqueeze(1)


        # Gaps after each token in S_hard
        if L_s > 0:
            internal_gaps = self.gap_internal_projector(S_hard_embeds) # (B, L_s, H_gap)
            h_gap = torch.cat([bos_gaps, internal_gaps], dim=1) # (B, L_s + 1, H_gap)
        else: # S_hard was empty, only BOS gap
            h_gap = bos_gaps

        return h_gap

