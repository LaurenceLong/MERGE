import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

def build_custom_tokenizer(
    tokenizer_name_or_path: str = 'gpt2', # 例如 'gpt2', 'llama-tokenizer-path'
    save_dir: str = "tokenizer_merge_custom",
    # vocab_size: int = 50257, # 如果要训练新tokenizer，可以指定
    # train_files: List[str] = None, # 训练tokenizer用的语料文件
    # add_special_tokens: Dict[str, str] = None, # 额外特殊token
) -> PreTrainedTokenizerBase:
    """
    构建或加载一个tokenizer。
    如果`save_dir`已存在且包含tokenizer文件，则从该目录加载。
    否则，从`tokenizer_name_or_path`初始化。

    注意: "从头训练" LLaMA2 通常意味着也从头训练一个适配其数据的SentencePiece tokenizer。
    此函数目前主要处理加载或基于现有结构初始化。
    若要训练全新tokenizer，需要使用 `tokenizers` 库并提供语料。
    """
    save_path = Path(save_dir)
    if save_path.exists() and (save_path / "tokenizer_config.json").exists():
        logger.info(f"从缓存目录加载tokenizer: {save_path.as_posix()}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(save_path.as_posix(), trust_remote_code=True)
        except Exception as e:
            logger.warning(f"从 {save_path} 加载tokenizer失败: {e}. 尝试从 {tokenizer_name_or_path} 重新初始化。")
            tokenizer = None # 标记为失败，以便后续重新初始化
    else:
        tokenizer = None

    if tokenizer is None:
        logger.info(f"从 {tokenizer_name_or_path} 初始化tokenizer。")
        # 对于LLaMA，通常使用LlamaTokenizer或LlamaTokenizerFast
        # 如果 tokenizer_name_or_path 是一个已训练好的LLaMA tokenizer路径，这会正确加载
        # 如果是 'gpt2' 等，它会加载对应的tokenizer结构
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        except Exception as e:
            logger.error(f"从 {tokenizer_name_or_path} 初始化tokenizer失败: {e}")
            logger.info("尝试使用通用的 'gpt2' tokenizer 作为后备结构。")
            tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True) # 后备

        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_path.as_posix())
        logger.info(f"Tokenizer已保存到: {save_path.as_posix()}")

    # 确保有PAD和MASK token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info(f"PAD token未设置。使用EOS token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}) 作为PAD token。")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            logger.warning("PAD 和 EOS token均未设置。添加新的PAD token: '[PAD]'。")
            # 这会增加vocab_size，模型embedding层需要对应调整
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    logger.info(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")


    if tokenizer.mask_token is None:
        logger.warning("MASK token未设置。添加新的MASK token: '[MASK]'。")
        # 这会增加vocab_size
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    logger.info(f"MASK token: '{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")

    return tokenizer


def collate_fn_for_merge(
    batch_items: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    text_column_name: str = "text" # 数据集中包含文本的列名
) -> Dict[str, torch.Tensor]:
    """
    数据整理函数，用于将一批样本转换为模型输入格式。
    1. 从batch_items中提取文本。
    2. 使用tokenizer进行编码。
    3. padding到max_length。
    """
    texts_to_tokenize = []
    for item in batch_items:
        text_content = item.get(text_column_name)
        if isinstance(text_content, str) and text_content.strip():
            texts_to_tokenize.append(text_content.strip())
        else:
            # 对于空或无效文本，可以填充一个空字符串或特定标记
            # tokenizer通常能处理空字符串（例如，输出 BOS/EOS 或仅padding）
            texts_to_tokenize.append(tokenizer.eos_token if tokenizer.eos_token else "") # 使用EOS或空串作为占位
            logger.debug(f"警告: 遇到空或无效的文本项。使用 '{texts_to_tokenize[-1]}' 替代。原始项: {item}")

    if not texts_to_tokenize:
        logger.warning("collate_fn 收到空的待处理文本列表。")
        return {
            "input_ids": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.long)
        }

    # 使用tokenizer进行批量编码
    tokenized_output = tokenizer(
        texts_to_tokenize,
        padding='max_length',       # padding到指定的最大长度
        truncation=True,            # 如果文本超过max_length则截断
        max_length=max_length,
        return_attention_mask=True, # 返回attention_mask
        return_tensors='pt'         # 直接返回PyTorch tensors
    )

    # 确保返回的字典包含模型期望的键
    batch_dict = {
        "input_ids": tokenized_output.input_ids,
        "attention_mask": tokenized_output.attention_mask
    }

    return batch_dict

