import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from transformers import PreTrainedModel  # 用于利用 save_pretrained, from_pretrained 等

from .configs import MERGEModelConfig
from .utils import gumbel_noise, linear_anneal
from .llama_components import LlamaDecoder, LlamaEncoder  # 你的LLaMA组件

logger = logging.getLogger(__name__)


class MERGELanguageModel(PreTrainedModel):
    config_class = MERGEModelConfig  # 关联自定义配置

    def __init__(self, config: MERGEModelConfig):
        super().__init__(config)  # 将config传递给父类
        self.config = config  # 保存配置副本

        # 1. 共享的词嵌入层 (Word Token Embeddings)
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id  # 设置padding_idx，使其在计算中被忽略（如果适用）
        )

        # 2. Decoder-1: Mask-Picker (用于删除阶段)，开启因果mask
        # 此Decoder用于选择哪些token被保留（构成骨架）
        self.mask_picker_decoder = LlamaDecoder(config, causal=True)  # 使用主config或修改后的config
        self.keep_logits_head = nn.Linear(config.hidden_size, 1)  # 输出每个token被保留的logit

        # 3. Decoder-2: Pointer-Inserter (用于插空阶段)，开启因果mask
        # 此Decoder用于在骨架的基础上，决定在哪些"缝隙"中插入MASK token
        self.pointer_inserter_decoder = LlamaDecoder(config, causal=True)
        # 输出维度: max_position_embeddings (代表可能的插入位置/缝隙数量) + 1 (可能代表"不在此处插入"或缝隙长度)
        # 原始代码是 cfg.max_len + 1，这里对应 config.max_position_embeddings + 1
        self.pointer_logits_head = nn.Linear(config.hidden_size, config.max_position_embeddings + 1)

        # 4. 主Encoder: Masked Language Modeling (使用LlamaEncoder结构)
        # 这个Encoder将从头开始训练，用于对损坏的输入进行MLM预测
        self.mlm_encoder = LlamaEncoder(config)  # LlamaEncoder内部应包含其LM head

        # 5. 权重绑定 (Weight Tying)
        # - 将主模型的词嵌入层与MLM Encoder的输出LM head的权重绑定
        if hasattr(self.mlm_encoder, 'lm_head') and self.mlm_encoder.lm_head is not None:
            self.mlm_encoder.lm_head.weight = self.word_embeddings.weight
        else:
            logger.warning("MLM Encoder没有lm_head属性或lm_head为None，无法进行输出层权重绑定。")

        # 6. MLM损失函数
        # ignore_index 设置为PAD token的ID，这样在计算损失时会忽略它们
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

        # 初始化权重 (PreTrainedModel会调用_init_weights)
        # self.post_init() # PreTrainedModel v4.X.X 之后推荐使用 post_init 进行最终处理
        self.init_weights()

    def _init_weights(self, module):
        """ 初始化模块的权重 (Hugging Face PreTrainedModel的默认实现会调用这个) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
            self,
            input_ids: torch.LongTensor,  # (B, L) 原始输入token IDs
            attention_mask: Optional[torch.Tensor] = None,  # (B, L) 原始输入的attention mask
            current_train_step: Optional[int] = None  # 当前训练步数，用于退火
    ) -> Dict[str, Any]:
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # --- 0. 确保提供了current_train_step (如果处于训练模式且需要退火) ---
        if self.training and current_train_step is None:
            logger.warning("current_train_step未提供，退火参数将使用最终值。")
            current_train_step = self.config.max_train_steps  # 设为最大步数以获取end_value

        # --- 1. 计算当前退火参数 ---
        current_temperature = linear_anneal(
            current_train_step,
            self.config.temperature_start, self.config.temperature_end,
            self.config.max_train_steps, self.config.warmup_steps
        )
        target_keep_ratio = linear_anneal(
            current_train_step,
            self.config.keep_ratio_start, self.config.keep_ratio_end,
            self.config.max_train_steps, self.config.warmup_steps
        )

        # --- 2. 获取输入嵌入 ---
        # (B, L, H)
        input_embeddings = self.word_embeddings(input_ids)

        # --- 3. Decoder-1: Mask Picker ---
        # 输入是原始序列的嵌入，输出是每个token被保留的logits
        # LlamaDecoder的forward期望 (target_embeds, memory_embeds, target_attention_mask, memory_attention_mask)
        # 原始代码: self.dec1(emb, emb) -> target=emb, memory=emb
        # 这意味着在mask_picker_decoder中，query, key, value 都来自 input_embeddings (即自注意力)
        # attention_mask 用于在自注意力中屏蔽padding token
        mask_picker_hidden_states = self.mask_picker_decoder(
            target_embeds=input_embeddings,
            # memory_embeds=input_embeddings, # 如果LlamaDecoder需要显式memory且用于自注意力
            attention_mask=attention_mask
        )  # (B, L, H)
        keep_logits = self.keep_logits_head(mask_picker_hidden_states).squeeze(-1)  # (B, L)

        # 使用Gumbel-Sigmoid (或Gumbel-Softmax的二分类形式) 进行软选择
        # soft_keep_probs 是每个token被保留的概率 (经过Gumbel扰动和sigmoid)
        gumbel_for_keep = gumbel_noise(keep_logits.shape, device)
        soft_keep_probs = torch.sigmoid((keep_logits + gumbel_for_keep) / current_temperature)  # (B, L)

        # 为了构建骨架，前向传播时使用硬选择 (基于阈值0.5)
        # 但梯度会通过soft_keep_probs传播 (Straight-Through Estimator的一种形式)
        hard_keep_mask = (soft_keep_probs > 0.5).bool()  # (B, L)

        # --- 4. 构建骨架序列 (Skeleton Construction) ---
        # 根据hard_keep_mask从input_ids中提取骨架token
        skeleton_input_ids_list = []
        skeleton_lengths = []
        max_skeleton_len_in_batch = 0

        for b_idx in range(batch_size):
            # 仅在非padding部分选择token
            actual_hard_keep_mask = hard_keep_mask[b_idx]
            if attention_mask is not None:
                actual_hard_keep_mask = actual_hard_keep_mask & attention_mask[b_idx].bool()

            kept_indices = actual_hard_keep_mask.nonzero(as_tuple=False).squeeze(-1)

            # 防止骨架为空，如果为空，至少保留第一个（非pad）token
            if kept_indices.numel() == 0:
                if attention_mask is not None:
                    first_valid_token_idx = attention_mask[b_idx].nonzero(as_tuple=False)
                    if first_valid_token_idx.numel() > 0:
                        kept_indices = first_valid_token_idx[0]
                    else:  # 如果整个序列都是padding（不太可能，但做防御）
                        kept_indices = torch.tensor([0], device=device, dtype=torch.long)
                else:
                    kept_indices = torch.tensor([0], device=device, dtype=torch.long)

            current_skeleton_ids = input_ids[b_idx, kept_indices]
            skeleton_input_ids_list.append(current_skeleton_ids)
            current_skel_len = len(current_skeleton_ids)
            skeleton_lengths.append(current_skel_len)
            if current_skel_len > max_skeleton_len_in_batch:
                max_skeleton_len_in_batch = current_skel_len

        # 将骨架序列padding到当前batch中的最大骨架长度
        padded_skeleton_ids = torch.full(
            (batch_size, max_skeleton_len_in_batch),
            fill_value=self.config.pad_token_id,  # 使用配置中的pad_token_id
            dtype=torch.long,
            device=device
        )
        # 为padding后的骨架创建attention_mask
        skeleton_attention_mask = torch.zeros(
            (batch_size, max_skeleton_len_in_batch), dtype=torch.long, device=device
        )

        for b_idx, skel_ids_item in enumerate(skeleton_input_ids_list):
            length = skel_ids_item.size(0)
            if length > 0:  # 只有当骨架非空时才填充
                padded_skeleton_ids[b_idx, :length] = skel_ids_item
                skeleton_attention_mask[b_idx, :length] = 1

        # 获取骨架的嵌入
        skeleton_embeddings = self.word_embeddings(padded_skeleton_ids)  # (B, max_skel_L, H)

        # --- 5. Decoder-2: Pointer-Inserter ---
        # 输入是骨架的嵌入，输出是指针logits，指示在骨架的哪些"缝隙"中插入MASK
        # 原始代码: self.dec2(sk_emb, sk_emb) -> 对骨架嵌入进行自注意力
        pointer_decoder_hidden_states = self.pointer_inserter_decoder(
            target_embeds=skeleton_embeddings,
            # memory_embeds=skeleton_embeddings, # 如果需要显式memory
            attention_mask=skeleton_attention_mask  # 屏蔽骨架中的padding token
        )  # (B, max_skel_L, H)
        # (B, max_skel_L, max_pos_embed + 1)
        pointer_logits = self.pointer_logits_head(pointer_decoder_hidden_states)

        # 对指针logits应用Gumbel-Softmax，得到在何处插入的软概率
        # softmax作用在最后一个维度 (max_position_embeddings + 1)
        gumbel_for_ptr = gumbel_noise(pointer_logits.shape, device)
        # (B, max_skel_L, max_pos_embed + 1)
        pointer_probs = F.softmax((pointer_logits + gumbel_for_ptr) / current_temperature, dim=-1)

        # --- 6. 构建损坏的输入 (Corrupted Input Construction for MLM Encoder) ---
        # 这是模型中最复杂的部分之一，原始代码使用了Python循环。
        # 理想情况下，这部分应该被向量化以提高效率，但为了保持逻辑一致性，
        # 我们首先尝试复现原始的逐样本循环逻辑。
        # 注意：这在GPU上会很慢。

        mask_token_id_to_insert = self.config.mask_token_id
        if mask_token_id_to_insert == -1 or mask_token_id_to_insert is None:
            raise ValueError("Config中的mask_token_id未正确设置。")

        corrupted_input_ids_batch_list = []
        for b_idx in range(batch_size):
            current_unpadded_skeleton_ids = skeleton_input_ids_list[b_idx]  # (current_skel_L,)
            current_skeleton_len = current_unpadded_skeleton_ids.size(0)

            # a. 决定插入多少个MASK token
            # 目标是使损坏序列的长度等于原始序列长度 `seq_length`
            num_masks_to_insert = seq_length - current_skeleton_len
            num_masks_to_insert = max(0, num_masks_to_insert)  # 不能是负数

            # b. 决定在哪里插入这些MASK token
            # 原始代码: `ptr_prob[b, :sk_lengths[b], :max_gap].mean(0)`
            # `max_gap` 在原始代码中是 `L + 1` (即 `seq_length + 1`)
            # `pointer_probs` 的形状是 (B, max_skel_L, max_pos_embed + 1)
            # 我们需要从 `pointer_probs` 中为当前样本选择插入位置
            if current_skeleton_len > 0:
                # 取当前骨架有效长度部分的pointer_probs，并在骨架长度维度上平均
                # `max_pos_embed + 1` 对应原始的 `max_gap`
                avg_insertion_slot_probs = pointer_probs[b_idx, :current_skeleton_len, :].mean(
                    dim=0)  # (max_pos_embed + 1,)
            else:  # 如果骨架为空（理论上已处理，但做防御）
                avg_insertion_slot_probs = torch.zeros(self.config.max_position_embeddings + 1, device=device)

            # c. 选择top-k个插入位置
            # `num_masks_to_insert` 是要插入的 MASK 数量
            # `avg_insertion_slot_probs` 是每个"缝隙"的得分
            # 缝隙数量是 `max_position_embeddings + 1`
            num_available_slots = avg_insertion_slot_probs.size(0)
            actual_num_masks_to_insert = min(num_masks_to_insert, num_available_slots)  # 不能插入超过可用缝隙的数量

            if actual_num_masks_to_insert > 0:
                # `torch.topk` 返回 (values, indices)
                # 我们需要 indices 来标记哪些缝隙被选中用于插入
                _, top_k_insertion_slot_indices = torch.topk(avg_insertion_slot_probs, k=actual_num_masks_to_insert)
            else:
                top_k_insertion_slot_indices = torch.tensor([], device=device, dtype=torch.long)

            # d. 构建损坏序列 (与原始代码的循环逻辑类似)
            # `is_insertion_slot` 标记哪些缝隙（slot_idx从0到max_pos_embed）需要插入MASK
            is_insertion_slot_flags = torch.zeros(num_available_slots, device=device, dtype=torch.bool)
            if actual_num_masks_to_insert > 0:
                is_insertion_slot_flags[top_k_insertion_slot_indices] = True

            reconstructed_sequence_for_item = []
            skeleton_token_iterator = 0
            # 遍历所有可能的"缝隙"位置 (0 to max_pos_embed)
            # 每个缝隙后可能跟一个骨架token
            for slot_idx in range(num_available_slots):  # num_available_slots == max_pos_embed + 1
                if len(reconstructed_sequence_for_item) >= seq_length:
                    break  # 已达到目标序列长度

                # 检查当前缝隙是否需要插入MASK
                if is_insertion_slot_flags[slot_idx]:
                    reconstructed_sequence_for_item.append(mask_token_id_to_insert)

                if len(reconstructed_sequence_for_item) >= seq_length:
                    break

                # 尝试放置下一个骨架token
                if skeleton_token_iterator < current_skeleton_len:
                    reconstructed_sequence_for_item.append(
                        current_unpadded_skeleton_ids[skeleton_token_iterator].item())
                    skeleton_token_iterator += 1

            # 将列表转换为tensor，并确保长度为seq_length (截断或填充)
            if len(reconstructed_sequence_for_item) > seq_length:
                corrupted_ids_tensor = torch.tensor(reconstructed_sequence_for_item[:seq_length], device=device,
                                                    dtype=torch.long)
            else:
                corrupted_ids_tensor = torch.tensor(reconstructed_sequence_for_item, device=device, dtype=torch.long)
                # 如果长度不足，用PAD token填充
                if corrupted_ids_tensor.size(0) < seq_length:
                    num_padding_needed = seq_length - corrupted_ids_tensor.size(0)
                    padding_tensor = torch.full((num_padding_needed,), self.config.pad_token_id, device=device,
                                                dtype=torch.long)
                    corrupted_ids_tensor = torch.cat([corrupted_ids_tensor, padding_tensor], dim=0)

            corrupted_input_ids_batch_list.append(corrupted_ids_tensor)

        # 将batch中所有损坏序列堆叠起来
        # (B, L)
        corrupted_batch_input_ids = torch.stack(corrupted_input_ids_batch_list, dim=0)
        # 为损坏的输入创建attention_mask (1表示非PAD token, 0表示PAD token)
        corrupted_batch_attention_mask = (corrupted_batch_input_ids != self.config.pad_token_id).long()

        # --- 7. MLM Encoder 前向传播 ---
        # a. 获取损坏输入的嵌入
        corrupted_input_embeddings = self.word_embeddings(corrupted_batch_input_ids)  # (B, L, H)

        # b. 通过MLM Encoder获取logits
        # LlamaEncoder的forward应返回一个包含"logits"键的字典
        mlm_encoder_outputs = self.mlm_encoder(
            input_embeds=corrupted_input_embeddings,
            attention_mask=corrupted_batch_attention_mask  # Encoder需要知道哪些是padding
        )
        mlm_logits = mlm_encoder_outputs["logits"]  # (B, L, VocabSize)

        # --- 8. 计算损失 ---
        # a. 重建损失 (Reconstruction Loss / MLM Loss)
        #    目标是让MLM Encoder预测出原始的 `input_ids`
        #    `mlm_loss_fn` 的 `ignore_index` 会处理原始 `input_ids` 中的padding
        loss_reconstruction = self.mlm_loss_fn(
            mlm_logits.view(-1, self.config.vocab_size),  # (B*L, VocabSize)
            input_ids.view(-1)  # (B*L,)
        )

        # b. 压缩正则化损失 (Compression Regularization Loss, L_comp)
        #    鼓励实际保留比例 (actual_keep_ratio) 接近目标保留比例 (target_keep_ratio)
        #    `soft_keep_probs` (B,L) 是保留概率，只在非padding部分计算平均值
        if attention_mask is not None:
            # `valid_tokens_mask` (B,L) 标记原始输入中的非padding token
            valid_tokens_mask_for_keep_ratio = attention_mask.bool()
            # `actual_keep_ratio` 是一个标量
            sum_soft_keep_probs_on_valid = (soft_keep_probs * valid_tokens_mask_for_keep_ratio).sum()
            num_valid_tokens = valid_tokens_mask_for_keep_ratio.sum().clamp(min=1)  # clamp避免除以0
            actual_keep_ratio = sum_soft_keep_probs_on_valid / num_valid_tokens
        else:  # 如果没有attention_mask，假设所有token都有效
            actual_keep_ratio = soft_keep_probs.mean()

        loss_compression = self.config.lambda_comp * F.relu(actual_keep_ratio - target_keep_ratio)

        # c. 指针正则化损失 (Pointer Regularization Loss, L_ptr)
        #    鼓励指针概率分布更集中 (低熵)
        #    原始公式: `L_ptr = cfg.l_ptr * (-ptr_prob.mean(-1).log().mean())`
        #    `ptr_prob` 是 (B, max_skel_L, max_pos_embed + 1)
        #    `ptr_prob.mean(-1)` 是 (B, max_skel_L), 对每个(batch, skel_token)的指针分布求平均概率
        #    然后取log，再对所有(batch, skel_token)求平均，再取负。
        #    这个公式有点不寻常，通常熵是 -sum(p * log p)。这里是 -log(mean(p))。
        #    我们会遵循原始公式，并确保只在有效骨架token上计算。

        # `skeleton_attention_mask` (B, max_skel_L) 标记骨架中的非padding token
        mean_probs_per_pointer_dist = pointer_probs.mean(dim=-1)  # (B, max_skel_L)
        log_mean_probs = torch.log(mean_probs_per_pointer_dist + 1e-9)  # 加epsilon防log(0)

        if skeleton_attention_mask.sum() > 0:
            # 只对有效骨架token的log_mean_probs求和并平均
            masked_log_mean_probs = log_mean_probs * skeleton_attention_mask  # 元素乘法，无效部分变0
            sum_masked_log_mean_probs = masked_log_mean_probs.sum()
            num_valid_skeleton_tokens = skeleton_attention_mask.sum().clamp(min=1)
            avg_neg_log_mean_prob = - (sum_masked_log_mean_probs / num_valid_skeleton_tokens)
        else:  # 如果骨架完全为空 (理论上已避免)
            avg_neg_log_mean_prob = torch.tensor(0.0, device=device)

        loss_pointer = self.config.lambda_ptr * avg_neg_log_mean_prob

        # d. 总损失
        total_loss = loss_reconstruction + loss_compression + loss_pointer

        # --- 9. 构建输出字典 (遵循原始模板) ---
        output_dict = {
            "loss": total_loss,
            "recon": loss_reconstruction.detach(),  # 使用 .detach() 获取无梯度版本用于记录
            "comp": loss_compression.detach(),
            "ptr": loss_pointer.detach(),
            "r_keep": actual_keep_ratio.detach(),  # 实际保留比例
            "temp": torch.tensor(current_temperature, device=device),  # 当前温度
            # 可以添加其他有用的指标
            "metrics_target_keep_ratio": torch.tensor(target_keep_ratio, device=device),
        }
        return output_dict

    # 如果需要一个 `generate` 方法 (例如用于demo中的文本生成)
    # 对于MLM模型，`generate` 通常指“填空”而不是自回归生成。
    # Hugging Face的 `AutoModelForMaskedLM` 也没有标准的 `generate` 方法。
    # 原始demo中的 `self.encoder.generate` 可能是一个特定用途的函数或简化。
    # 如果你的 LlamaEncoder (MLM编码器) 要支持某种形式的生成，你需要单独实现。
    # 例如，迭代地预测 [MASK] token。
    # def generate_mlm_fill(self, input_ids_with_masks, max_iterations=10, ...):
    #     # ... 实现填空逻辑 ...
    #     pass

