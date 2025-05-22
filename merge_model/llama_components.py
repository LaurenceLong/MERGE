"""
merge/llama_components.py

实现了完整的 LLaMA 模块：
  - LlamaAttention: 多头注意力（可扩展接入 RoPE）
  - LlamaMLP: 基于 SwiGLU 设计的前馈网络
  - RMSNorm: RMS 标准化（LLaMA 风格）
  - LlamaLayer: 包含注意力和 MLP 的 Transformer 层（带残差与 RMSNorm）
  - LlamaStack: 堆叠多个 LlamaLayer，并在末尾加 RMSNorm
  - LlamaEncoder: 模拟 Encoder 结构（依然利用 LlamaStack + LM head）
  - LlamaDecoder: 类似 Decoder 的结构（默认开启因果mask，用于保证自注意力只能关注历史信息）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        # 若需要，可在此处添加 Rotary Positional Embeddings 的实现

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            encoder_hidden_states: torch.Tensor = None,
            encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            query = self.q_proj(hidden_states)
            key = self.k_proj(encoder_hidden_states)
            value = self.v_proj(encoder_hidden_states)
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

        B, seq_len, _ = query.size()
        query = query.view(B, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # 如果传入的 attention_mask 维度为2，则扩展为 (B,1,1,seq_len)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores + (1.0 - attention_mask) * -10000.0

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_size)
        output = self.out_proj(context)
        return output


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size * 2)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.fc1(hidden_states)
        gate, x = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.fc2(x)
        return x


class LlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.dropout_prob,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 自注意力部分
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + attn_output

        # MLP 部分
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states


class LlamaStack(nn.Module):
    def __init__(self, config, num_layers_override: int = None):
        super().__init__()
        num_layers = num_layers_override if num_layers_override is not None else config.num_hidden_layers_default
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaEncoder(nn.Module):
    """
    Encoder 实现采用与 Decoder 类似的堆叠结构，
    但用于 Masked Language Modeling 的填充任务（输出 logits）。
    """
    def __init__(self, config, num_layers_override: int = None):
        super().__init__()
        num_layers = num_layers_override if num_layers_override is not None else config.num_hidden_layers_encoder
        self.stack = LlamaStack(config, num_layers)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_embeds: torch.Tensor, attention_mask: torch.Tensor = None) -> dict:
        hidden_states = self.stack(hidden_states=input_embeds, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return {"hidden_states": hidden_states, "logits": logits}

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        pass


class LlamaDecoder(nn.Module):
    """
    Decoder 实现，适用于 Mask-Picker 模块及其他解码场景。
    与 Encoder 类似，但不附加 LM head。
    默认开启因果（causal）mask，用于保证自注意力中只关注历史token。
    """
    def __init__(self, config, causal: bool = True, num_layers_override: int = None):
        super().__init__()
        num_layers = num_layers_override if num_layers_override is not None else config.num_hidden_layers_encoder
        self.stack = LlamaStack(config, num_layers)
        self.causal = causal

    def forward(
        self,
        target_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.causal:
            B, L, _ = target_embeds.shape
            device = target_embeds.device
            # 生成因果mask（下三角矩阵），类型与输入保持一致
            causal_mask = torch.tril(torch.ones((L, L), device=device, dtype=target_embeds.dtype))
            if attention_mask is None:
                attn_mask = torch.ones((B, L), device=device, dtype=target_embeds.dtype)
            else:
                attn_mask = attention_mask
            # 将 attention_mask 扩展至 (B, L, L)
            combined_mask = attn_mask.unsqueeze(1) * causal_mask  # (B, L, L)
            # 再扩展一个维度得到 (B, 1, L, L)
            final_mask = combined_mask.unsqueeze(1)
            attention_mask = final_mask
        output = self.stack(
            hidden_states=target_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return output