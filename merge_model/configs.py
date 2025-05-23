from dataclasses import dataclass, field
from transformers import PretrainedConfig

@dataclass
class MERGEModelConfig(PretrainedConfig):
    model_type: str = "merge_mdlm_llama"

    # LLAMA架构参数
    vocab_size: int = 50257
    hidden_size: int = 768
    intermediate_size: int = field(default_factory=lambda: 768 * 4)
    num_hidden_layers_decoder: int = 6  # For LLAMAdec in M1
    num_hidden_layers_encoder: int = 12 # For E_MDLM
    num_hidden_layers_gap_encoder: int = 2 # For M2's GapEncoder (example)
    num_attention_heads: int = 12
    max_seq_len: int = 256
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5

    # E_MDLM specific
    rope_theta: float = 10000.0 # For RoPE, if implemented

    # Training parameters
    learning_rate: float = 3e-4 # General learning rate
    learning_rate_gate: float = None # Specific LR for Gate
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps_ratio: float = 0.05 # Used in train_end2end.py for scheduler
    max_train_steps: int = 100_000 # Total training steps for annealing etc.
    gradient_clip_val: float = 1.0

    # MERGE × MDLM Scheme Parameters
    # General
    max_rounds_l: int = 6  # L_max in scheme

    # Gate Parameters
    gate_mlp_hidden_dim: int = 64 # Hidden dim for 2-layer MLP in Gate
    gate_feature_dim: int = 0 # Will be determined by concat of features
    round_emb_dim: int = 32
    gate_tau_start: float = 2.0
    gate_tau_end: float = 0.2
    gate_tau_anneal_steps: int = 5000 # Example: 5 epochs if 1 epoch = 1000 steps
    gate_low_conf_threshold_nats: float = 1.5 # τ_H for low_conf_ratio

    # Mask-Picker (M₁) Parameters
    m1_ffn_intermediate_dim: int = field(default_factory=lambda: 768 * 2) # For FFN in M1 after LLAMAdec
    k_del_ratio: float = 0.2
    k_del_min: int = 1
    k_del_max: int = 20
    m1_tau_del_start: float = 2.0 # τ_del
    m1_tau_del_end: float = 0.2
    m1_tau_del_anneal_steps: int = 5000 # Example, sync with gate_tau_anneal_steps

    # Mask-Inserter (M₂) Parameters
    k_ins_target_ratio: float = 0.2 # Ratio for |Y*| - |S|
    k_ins_min: int = 1
    k_ins_max: int = 20
    max_masks_per_insertion_round_m2: int = 8 # For stable Multinomial_ST (Tip 3)
    gap_encoder_hidden_dim: int = hidden_size # Hidden dim for GapEncoder output

    # Loss Weights (λ values)
    lambda_gate_sparsity: float = 0.1   # λ_s
    lambda_gate_reward: float = 0.2     # λ_c (reward for "sentence maturity improvement")
    lambda_m1_entropy: float = 0.05     # λ_comp (KL(p_del || Ber(r_target)))
    lambda_m2_entropy: float = 0.01     # λ_ins (Entropy(z_g))
    
    # Target for M₁ KL divergence (r_target in Ber(r_target))
    # This is the desired average deletion proportion for M1
    m1_target_deletion_ratio: float = 0.15 # Example: Corresponds to K_del_ratio, but for soft probs

    # MDLM specific weighting for L_recon
    # alpha_mdlm_t_factor: float = 1.0 # Factor for cos^2(pi*t/2) if needed, usually 1.0

    # Tokenizer and Paths
    tokenizer_name_or_path: str = 'gpt2'
    tokenizer_save_dir: str = "tokenizer_merge_mdlm"
    pad_token_id: int = -100 # Will be set by tokenizer
    mask_token_id: int = -1  # Will be set by tokenizer
    eos_token_id: int = -1 # Will be set
    bos_token_id: int = -1 # Will be set

    # Teacher Forcing
    teacher_forcing_warmup_epochs: int = 10 # Epochs to use Y*
    teacher_forcing_prob_decay_start_epoch: int = 10 # Epoch to start decaying Y* prob
    teacher_forcing_prob_end_epoch: int = 20 # Epoch by which Y* prob is 0 (or min)
    teacher_forcing_initial_prob: float = 1.0
    teacher_forcing_final_prob: float = 0.0 # Probability of using Y* after decay

    # Other
    seed: int = 42
    log_every_steps: int = 100
    checkpoint_dir: str = "checkpoints_merge_mdlm"
    initializer_range: float = 0.02
    pruned_heads: dict = field(default_factory=dict)

    # Pseudo-time condition for E_MDLM
    pseudo_time_emb_dim: int = 64 # Embedding dim for mask_ratio

    def __init__(self, **kwargs):
        known_keys = list(self.__dataclass_fields__.keys())
        our_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in known_keys}
        super().__init__(**kwargs)
        for k, v in our_kwargs.items():
            setattr(self, k, v)
        self.__post_init__()

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        if self.m1_ffn_intermediate_dim is None:
            self.m1_ffn_intermediate_dim = self.hidden_size * 2
        
        # Calculate gate_feature_dim based on expected features if possible,
        # or set it to a reasonable estimate.
        # Features: H̄, margin, low_conf_ratio, ppl_self, round_emb
        # Approx: 1 (entropy) + 1 (margin) + 1 (low_conf) + 1 (ppl) + round_emb_dim
        self.gate_feature_dim = 4 + self.round_emb_dim


# Example:
# config = MERGEModelConfig(vocab_size=32000, max_train_steps=50000)
# print(config)
