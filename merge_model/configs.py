from dataclasses import dataclass, field
from transformers import PretrainedConfig # 保持与HuggingFace生态的兼容性

@dataclass
class MERGEModelConfig(PretrainedConfig):
    """
    MERGELanguageModel的配置类。
    继承自PretrainedConfig以利用其save_pretrained/from_pretrained等功能。
    """
    model_type: str = "merge_custom_llama"

    # LLAMA架构参数 (你需要根据你的LLAMA实现调整这些名称和值)
    vocab_size: int = 50257
    hidden_size: int = 768
    intermediate_size: int = field(default_factory=lambda: 768 * 4) # LLaMA中通常是 (2/3 * 4 * H) * 2 / 2 = 1.33 * 4 * H, 调整这里
    num_hidden_layers_decoder: int = 6  # 用于 dec1 (mask_picker) 和 dec2 (pointer_inserter)
    num_hidden_layers_encoder: int = 6  # 用于主 MLM encoder
    num_hidden_layers_default: int = 6  # 默认的decoder/encoder层数
    num_attention_heads: int = 12
    max_seq_len: int = 256              # 最大序列长度
    dropout_prob: float = 0.1           # Transformer层中的dropout概率
    layer_norm_eps: float = 1e-5        # LayerNorm (或RMSNorm) 的 epsilon

    # 训练参数
    learning_rate_main: float = 1e-4
    learning_rate_encoder: float = 3e-5 # 如果encoder部分使用不同学习率
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps: int = 1000 # 注意：train.py 中使用的是 warmup_steps_ratio
    max_train_steps: int = 10_000

    # MERGE 特定参数
    keep_ratio_start: float = 0.7
    keep_ratio_end: float = 0.3
    temperature_start: float = 1.0      # Gumbel softmax 温度
    temperature_end: float = 0.1
    lambda_comp: float = 0.5            # 压缩损失权重
    lambda_ptr: float = 0.05            # 指针损失权重

    # Tokenizer 和路径参数
    tokenizer_name_or_path: str = 'gpt2' # 用于初始化tokenizer的名称或路径
    tokenizer_save_dir: str = "tokenizer_merge_custom" # 保存自定义tokenizer的目录
    pad_token_id: int = -100 # 将由tokenizer设置, -100常用于CrossEntropyLoss的ignore_index
    mask_token_id: int = -1  # 将由tokenizer设置

    # 其他参数
    seed: int = 42
    log_every_steps: int = 100
    checkpoint_dir: str = "checkpoints_merge_custom" # 保存模型检查点的目录
    initializer_range: float = 0.02 # 用于初始化权重的标准差

    # 添加缺失的属性以兼容 PreTrainedModel 的初始化流程
    pruned_heads: dict = field(default_factory=dict) # <--- 添加这一行

    # 添加这一行以捕获额外参数
    def __init__(self, **kwargs):
        # 首先从kwargs中提取我们已知的参数
        known_keys = list(self.__dataclass_fields__.keys())
        our_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in known_keys}

        # 调用父类的__init__来处理剩余的kwargs（如architectures等）
        super().__init__(**kwargs)

        # 设置我们自己的属性
        for k, v in our_kwargs.items():
            setattr(self, k, v)

        # 确保所有必需字段都有值
        self.__post_init__()

    def __post_init__(self):
        # 可以在这里进行一些参数的校验或后处理
        if self.intermediate_size is None:
            # LLaMA 2 的 intermediate_size 计算方式:
            # multiple_of = 256
            # ffn_dim_multiplier = 1.0 # or custom_ffn_dim_multiplier
            # self.intermediate_size = int(ffn_dim_multiplier * self.hidden_size * 4 * 2 / 3)
            # self.intermediate_size = multiple_of * ((self.intermediate_size + multiple_of - 1) // multiple_of)
            # 简化版：
            self.intermediate_size = self.hidden_size * 4


# 使用示例:
# config = MERGEModelConfig(vocab_size=32000, max_train_steps=50000)
# print(config)

