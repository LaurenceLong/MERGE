# inference.py
import argparse

import torch
from transformers import AutoTokenizer

from .model import MERGELanguageModel  # 确保 MERGE 在 PYTHONPATH


def main():
    parser = argparse.ArgumentParser(description="使用训练好的 MERGELanguageModel 进行推理。")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="包含模型权重、配置和tokenizer的已训练模型目录的路径 (例如 'outputs_merge_llama/final_model')"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is an example sentence to test the MERGE model.",
        help="用于推理的输入文本。"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备 (cuda or cpu)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,  # 应与训练时的 max_seq_len 类似
        help="输入文本的最大处理长度。"
    )
    args = parser.parse_args()

    # 1. 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print(f"Tokenizer loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("确保模型路径下包含 tokenizer.json 和其他必要的tokenizer文件。")
        return

    # 2. 加载模型配置 (可选，但可以用来验证)
    # model_config = MERGEModelConfig.from_pretrained(args.model_path)
    # print(f"Model config loaded. Vocab size: {model_config.vocab_size}")

    # 3. 加载模型
    # 确保 MERGELanguageModel 类已注册或可被 from_pretrained 找到
    # 如果遇到 AutoModel 无法找到自定义类的问题，直接使用 MERGELanguageModel.from_pretrained
    try:
        model = MERGELanguageModel.from_pretrained(args.model_path)
        model.to(args.device)
        model.eval()  # 设置为评估模式
        print(f"Model loaded successfully from {args.model_path} and moved to {args.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("确保 MERGE 包在PYTHONPATH中，并且模型路径正确。")
        return

    # 4. 准备输入
    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding="max_length"  # 或者根据需要调整 padding 策略
    )
    input_ids = inputs.input_ids.to(args.device)
    attention_mask = inputs.attention_mask.to(args.device)

    print(f"\nOriginal Text: {args.text}")
    print(f"Tokenized Input IDs: {input_ids[0, :attention_mask.sum()].tolist()}")  # 只显示非padding部分

    # 5. 执行推理
    with torch.no_grad():
        # 对于推理，current_train_step 应设置为一个较大的值，以使用退火后的参数
        # 例如，使用配置中的 max_train_steps
        inference_step = model.config.max_train_steps

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            current_train_step=inference_step
        )

    # 6. 解析输出
    # (前提：你已经按照上面的建议修改了 model.py 中的 forward 方法以返回这些键)
    if "corrupted_input_ids_for_eval" not in outputs or "mlm_logits_for_eval" not in outputs:
        print("\n[Warning] 'corrupted_input_ids_for_eval' or 'mlm_logits_for_eval' not found in model outputs.")
        print("This inference script expects MERGELanguageModel.forward() to return these keys in eval mode.")
        print("Please consider modifying model.py as suggested in the documentation.")

        # 备选方案：演示MLM Encoder的填空能力 (类似 train_end2end.py 中的demo)
        print("\nRunning a simpler MLM fill-in demo instead...")

        # 手动创建一些 MASK
        masked_input_ids = input_ids.clone()
        num_tokens = attention_mask.sum().item()
        if num_tokens > 5:  # 确保有足够token去mask
            # mask 掉大约 15% 的 token (跳过CLS/SEP等特殊token，如果适用)
            # 为简单起见，这里随机选择一些位置（不考虑是否为特殊token）
            mask_indices = torch.randperm(num_tokens - 2)[:max(1, int((num_tokens - 2) * 0.15))] + 1  # 避开头尾
            if mask_indices.numel() > 0:
                masked_input_ids[0, mask_indices] = tokenizer.mask_token_id
            else:  # 如果没有选中任何token（例如序列太短），就mask中间一个
                masked_input_ids[0, num_tokens // 2] = tokenizer.mask_token_id

            print(f"Manually Masked Input Text: {tokenizer.decode(masked_input_ids[0], skip_special_tokens=False)}")

            # 获取嵌入
            demo_word_embeds = model.word_embeddings(masked_input_ids)
            demo_input_embeds = model._apply_positional_embeddings(demo_word_embeds)

            # 通过MLM Encoder
            mlm_output = model.mlm_encoder(
                input_embeds=demo_input_embeds,
                attention_mask=attention_mask  # 使用原始的attention_mask
            )
            predicted_logits = mlm_output["logits"]
            predicted_token_ids_mlm_demo = torch.argmax(predicted_logits, dim=-1)

            # 仅填充被mask的位置
            reconstructed_ids_mlm_demo = masked_input_ids.clone()
            if tokenizer.mask_token_id in reconstructed_ids_mlm_demo:
                mask_positions = (reconstructed_ids_mlm_demo == tokenizer.mask_token_id).squeeze()
                reconstructed_ids_mlm_demo[0, mask_positions] = predicted_token_ids_mlm_demo[0, mask_positions]

            reconstructed_text_mlm_demo = tokenizer.decode(reconstructed_ids_mlm_demo[0, :num_tokens],
                                                           skip_special_tokens=True)
            print(f"MLM Reconstructed Text: {reconstructed_text_mlm_demo}")
        else:
            print("Input text too short for MLM fill-in demo.")
        return

    # --- 如果 model.py 已按建议修改，则执行以下 ---
    corrupted_ids = outputs["corrupted_input_ids_for_eval"]
    mlm_logits = outputs["mlm_logits_for_eval"]

    # 获取模型生成的损坏文本
    # 注意：corrupted_ids 可能有不同的padding，需要根据其自身的attention mask（如果有）或有效长度来解码
    # 这里假设 corrupted_ids 与原始 input_ids 长度相同，且使用相同的 tokenizer.pad_token_id
    corrupted_attention_mask = (corrupted_ids != tokenizer.pad_token_id).long()
    num_corrupted_tokens = corrupted_attention_mask[0].sum().item()
    corrupted_text = tokenizer.decode(corrupted_ids[0, :num_corrupted_tokens], skip_special_tokens=False)  # 保留[MASK]等
    print(f"\nModel-Generated Corrupted Text (with [MASK]s): {corrupted_text}")

    # 从 MLM logits 预测 token IDs
    predicted_token_ids = torch.argmax(mlm_logits, dim=-1)  # (B, L)

    # 将预测的token填充到损坏序列的 [MASK] 位置，或者直接解码预测序列
    # 简单起见，我们直接解码模型对整个损坏序列的预测结果
    # (更精确的做法是只替换[MASK] token的预测)
    reconstructed_from_prediction_ids = predicted_token_ids.clone()

    # 一个更精确的重建方法：只替换掉corrupted_ids中的MASK token
    final_reconstructed_ids = corrupted_ids.clone()
    if tokenizer.mask_token_id is not None:
        mask_token_filter = (corrupted_ids == tokenizer.mask_token_id)
        final_reconstructed_ids[mask_token_filter] = predicted_token_ids[mask_token_filter]
    else:  # 如果没有mask_token_id的概念，则整个序列都是“预测”
        final_reconstructed_ids = predicted_token_ids

    reconstructed_text = tokenizer.decode(final_reconstructed_ids[0, :num_corrupted_tokens], skip_special_tokens=True)
    print(f"Model Reconstructed Text: {reconstructed_text}")

    # 打印一些具体位置的预测 (例如，被mask的位置)
    if tokenizer.mask_token_id is not None:
        mask_indices_in_corrupted = (corrupted_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False).squeeze(-1)
        if mask_indices_in_corrupted.numel() > 0:
            print("\n--- Predictions at MASKed positions ---")
            for idx in mask_indices_in_corrupted:
                original_token_at_mask = input_ids[0, idx].item()  # 原始token
                predicted_token_at_mask = predicted_token_ids[0, idx].item()  # 模型预测的token
                print(f"Position {idx.item()}: "
                      f"Original: {tokenizer.decode([original_token_at_mask])} ({original_token_at_mask}), "
                      f"Predicted: {tokenizer.decode([predicted_token_at_mask])} ({predicted_token_at_mask})")
        else:
            print("\nNo MASK tokens found in the model-generated corrupted sequence by the inference script's check.")


if __name__ == "__main__":
    main()

