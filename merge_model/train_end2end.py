import argparse
import json
import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torch.optim import AdamW  # PyTorch自带AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # 自动选择适合环境的tqdm (notebook, console)
from transformers import get_linear_schedule_with_warmup

# 从本地merge包导入组件
from .configs import MERGEModelConfig
from .data import build_custom_tokenizer, collate_fn_for_merge
from .model import MERGELanguageModel

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_training_arguments():
    parser = argparse.ArgumentParser(description="从头训练 MERGELanguageModel (LLaMA-based)。")

    # 数据集参数
    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Hugging Face数据集名称或本地路径。")
    parser.add_argument("--dataset_name", type=str, default="wikitext-2-v1", help="数据集的特定配置名称 (例如 'wikitext-103-raw-v1')。")
    parser.add_argument("--text_column", type=str, default="text", help="数据集中包含文本的列名。")
    parser.add_argument("--validation_split_percentage", type=int, default=5, help="从训练集中划分多少百分比作为验证集（如果没有预设验证集）。")


    # 模型配置参数 (允许命令行覆盖部分默认值)
    # MERGEModelConfig中定义的参数都可以通过命令行传入，例如 --vocab_size 50000
    # 这里只列举几个示例，实际可以通过 **vars(args) 传递给config
    parser.add_argument("--max_seq_len", type=int, help="模型最大序列长度。")

    # 训练过程参数
    parser.add_argument("--max_train_steps", type=int, default=10000, help="总训练步数。")
    parser.add_argument("--batch_size_train_per_device", type=int, default=32, help="每个GPU/CPU的训练批次大小。")
    parser.add_argument("--learning_rate_main", type=float, help="主模型部分的学习率。")
    parser.add_argument("--learning_rate_encoder", type=float, help="MLM Encoder部分的学习率 (如果不同)。")
    parser.add_argument("--weight_decay", type=float, help="AdamW优化器的权重衰减。")
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.05, help="学习率预热步数占总步数的比例。")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数。")


    # 路径和日志参数
    parser.add_argument("--output_dir", type=str, default="outputs_merge_llama", help="保存检查点、日志和最终模型的目录。")
    parser.add_argument("--tokenizer_save_dir", type=str, help="保存/加载tokenizer的特定目录 (默认在output_dir下)。")
    parser.add_argument("--log_every_steps", type=int, default=100, help="每N步记录一次训练信息。")
    parser.add_argument("--save_every_steps", type=int, default=1000, help="每N步保存一次模型检查点。")
    parser.add_argument("--seed", type=int, default=42, help="用于复现的随机种子。")
    parser.add_argument("--tensorboard_log_name", type=str, default="merge_llama_training_logs", help="Tensorboard日志的名称。")

    # 可以添加更多MERGEModelConfig中的参数，例如 --lambda_comp 0.6
    # 通过解析后，将vars(args)中与config字段匹配的项传递给config构造函数

    args = parser.parse_args()
    return args

def main():
    args = parse_training_arguments()

    # --- 1. 初始化 Accelerator ---
    # Accelerator会自动处理设备分配 (CPU/GPU/TPU)、分布式训练等
    # `logging_dir` 用于 TensorBoard 和其他日志
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, # 主输出目录
        logging_dir=Path(args.output_dir) / args.tensorboard_log_name
    )
    accelerator = Accelerator(
        log_with="tensorboard", # 启用tensorboard日志
        project_config=accelerator_project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" # 如果需要，可以启用混合精度训练
    )

    # 设置日志级别，确保只在主进程打印过多信息
    if accelerator.is_local_main_process:
        logging.basicConfig(level=logging.INFO) # 主进程显示INFO及以上
        logger.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING) # 其他进程只显示WARNING及以上
        logger.setLevel(logging.WARNING)

    logger.info(f"Accelerator state: {accelerator.state}")
    logger.info(f"Distributed type: {accelerator.distributed_type}")
    logger.info(f"Number of processes: {accelerator.num_processes}")

    # --- 2. 设置随机种子 ---
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"随机种子已设置为: {args.seed}")

    # --- 3. 加载或构建模型配置 ---
    # 从命令行参数更新配置默认值
    config_overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and hasattr(MERGEModelConfig, k)
    }
    model_config = MERGEModelConfig(**config_overrides)
    logger.info(f"模型配置参数: {model_config.to_json_string()}")


    # --- 4. 构建或加载Tokenizer ---
    tokenizer_actual_save_dir = args.tokenizer_save_dir or Path(args.output_dir) / "tokenizer"
    tokenizer = build_custom_tokenizer(
        tokenizer_name_or_path=model_config.tokenizer_name_or_path,
        save_dir=tokenizer_actual_save_dir,
    )
    # 更新config中的tokenizer相关ID (如果tokenizer是新建或修改过的)
    model_config.vocab_size = len(tokenizer) # add_special_tokens后需要通过len()获取正确词汇表大小
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.mask_token_id = tokenizer.mask_token_id

    if model_config.pad_token_id is None or model_config.mask_token_id is None:
        raise ValueError("Tokenizer必须包含PAD和MASK token，并且它们的ID已在config中设置。")
    logger.info(f"Tokenizer加载完毕。Vocab size: {model_config.vocab_size}, PAD ID: {model_config.pad_token_id}, MASK ID: {model_config.mask_token_id}")


    # --- 5. 加载和预处理数据集 ---
    logger.info(f"开始加载数据集: {args.dataset_path} (配置: {args.dataset_name})")
    raw_datasets = load_dataset(args.dataset_path, args.dataset_name, trust_remote_code=True)

    # 处理数据集划分 (训练集/验证集)
    if "validation" not in raw_datasets.keys() and args.validation_split_percentage > 0:
        logger.info(f"未找到'validation'集。从'train'集中划分 {args.validation_split_percentage}% 作为验证集。")
        # split_dataset返回一个DatasetDict
        train_validation_split = raw_datasets["train"].train_test_split(
            test_size=args.validation_split_percentage / 100.0,
            seed=args.seed
        )
        raw_datasets["train"] = train_validation_split["train"]
        raw_datasets["validation"] = train_validation_split["test"]
    elif "validation" not in raw_datasets.keys():
        logger.warning("数据集中没有验证集，并且未指定划分比例。将不使用验证集。")
        raw_datasets["validation"] = None # 显式设为None

    # 过滤空文本
    def filter_empty_texts_fn(example):
        return example[args.text_column] is not None and len(example[args.text_column].strip()) > 0

    with accelerator.main_process_first(): # 确保只有一个进程执行下载/预处理
        train_dataset = raw_datasets["train"].filter(filter_empty_texts_fn, num_proc=4 if accelerator.num_processes > 1 else 1)
        logger.info(f"训练集样本数 (过滤后): {len(train_dataset)}")
        if raw_datasets["validation"] is not None:
            eval_dataset = raw_datasets["validation"].filter(filter_empty_texts_fn, num_proc=4 if accelerator.num_processes > 1 else 1)
            logger.info(f"验证集样本数 (过滤后): {len(eval_dataset)}")
        else:
            eval_dataset = None

    # --- 6. 创建DataLoader ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_train_per_device,
        shuffle=True,
        collate_fn=lambda batch_items: collate_fn_for_merge(
            batch_items, tokenizer, model_config.max_seq_len, args.text_column
        ),
        num_workers=4, # 根据系统调整
        pin_memory=True if accelerator.device.type == 'cuda' else False,
    )
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size_train_per_device * 2, # 验证时batch可以大一些
            collate_fn=lambda batch_items: collate_fn_for_merge(
                batch_items, tokenizer, model_config.max_seq_len, args.text_column
            ),
            num_workers=4,
            pin_memory=True if accelerator.device.type == 'cuda' else False,
        )
    else:
        eval_dataloader = None


    # --- 7. 初始化模型 ---
    logger.info("从头初始化 MERGELanguageModel...")
    # 注意：这里是关键，MERGELanguageModel内部的LLaMA组件需要能正确地从头初始化
    model = MERGELanguageModel(model_config)
    logger.info(f"模型结构:\n{model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型可训练参数量: {num_params/1e6:.2f} M")


    # --- 8. 初始化优化器 ---
    # 将参数分组，可以为不同部分设置不同学习率
    encoder_param_names = ["mlm_encoder."] # MLM Encoder的参数通常以这个前缀开头
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in encoder_param_names) and p.requires_grad],
            "lr": args.learning_rate_main or model_config.learning_rate_main,
            "name": "main_model_parts"
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in encoder_param_names) and p.requires_grad],
            "lr": args.learning_rate_encoder or model_config.learning_rate_encoder or (args.learning_rate_main or model_config.learning_rate_main), # 如果未指定encoder_lr，则使用main_lr
            "name": "mlm_encoder_part"
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        weight_decay=args.weight_decay or model_config.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    logger.info(f"优化器 AdamW 配置完成。Main LR: {optimizer_grouped_parameters[0]['lr']:.2e}, Encoder LR: {optimizer_grouped_parameters[1]['lr']:.2e}")


    # --- 9. 初始化学习率调度器 ---
    num_warmup_steps = int(args.max_train_steps * args.warmup_steps_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    logger.info(f"学习率调度器配置完成。预热步数: {num_warmup_steps}")


    # --- 10. 使用Accelerator准备所有组件 ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if eval_dataloader:
        eval_dataloader = accelerator.prepare(eval_dataloader)


    # --- 11. 初始化TensorBoard tracker (如果使用) ---
    if accelerator.is_main_process:
        # `accelerator.init_trackers` 会在 `Accelerator` 初始化时自动调用（如果 `log_with` 指定了）
        # 这里可以用来记录超参数等额外信息
        # 获取TensorBoard tracker实例
        tb_tracker = accelerator.get_tracker("tensorboard")

        if tb_tracker is not None:
            # 使用 tracker 的 log_text 方法记录文本信息
            # log_text 方法接受一个字典，其中键是TensorBoard中的tag，值是要记录的文本
            # 这个日志记录发生在训练开始前，所以使用 step=0

            hyperparameters_text = json.dumps(vars(args), indent=2)
            tb_tracker.log(
                {"hyperparameters": hyperparameters_text},
                step=0
            )

            model_config_text = model_config.to_json_string(use_diff=False)
            tb_tracker.log(
                {"model_config": model_config_text},
                step=0
            )
            logger.info("已通过 TensorBoardTracker.log_text 记录超参数和模型配置。")
        else:
            logger.warning("TensorBoard tracker 未找到 (tb_tracker is None)，无法记录文本配置。")
        logger.info(f"TensorBoard 日志将保存在: {accelerator.logging_dir}")


    # --- 12. 训练循环 ---
    logger.info("***** 开始训练 *****")
    logger.info(f"  总训练步数 = {args.max_train_steps}")
    logger.info(f"  每设备批次大小 = {args.batch_size_train_per_device}")
    logger.info(f"  总即时批次大小 (跨设备和梯度累积) = {args.batch_size_train_per_device * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")

    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process, desc="训练进度")
    completed_steps = 0

    # 根据总步数计算大致的epoch数，或直接迭代直到步数完成
    # 如果数据集很大，可能不需要完成所有epoch
    for epoch in range(int(args.max_train_steps / len(train_dataloader)) + 1 if len(train_dataloader) > 0 else 1):
        model.train()
        total_loss_epoch = 0
        for step, batch in enumerate(train_dataloader):
            if completed_steps >= args.max_train_steps:
                break

            with accelerator.accumulate(model): # 处理梯度累积
                # 模型forward需要 current_train_step 用于退火
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    current_train_step=completed_steps
                )
                loss = outputs["loss"]
                total_loss_epoch += loss.detach().float() # 累积epoch损失用于平均

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            completed_steps += 1
            progress_bar.update(1)

            if completed_steps % args.log_every_steps == 0:
                avg_loss_recent_steps = total_loss_epoch / (step + 1) # 当前epoch内到目前为止的平均损失
                log_metrics = {
                    "train_loss": avg_loss_recent_steps.item(), # 或者用loss.item()记录当前step的loss
                    "learning_rate": lr_scheduler.get_last_lr()[0], # 获取当前学习率
                    "epoch": epoch + 1,
                    "step_in_epoch": step + 1,
                    "completed_steps": completed_steps,
                }
                # 添加模型输出中的其他指标
                for k, v in outputs.items():
                    if k != "loss" and isinstance(v, torch.Tensor):
                        log_metrics[k.replace("metrics_", "metric_")] = v.item()

                accelerator.log(log_metrics, step=completed_steps)
                progress_bar.set_postfix(train_loss=f"{log_metrics['train_loss']:.4f}", lr=f"{log_metrics['learning_rate']:.2e}")

            # 定期保存检查点
            if completed_steps % args.save_every_steps == 0 and completed_steps > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_path = Path(args.output_dir) / f"checkpoint_step_{completed_steps}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_path, safe_serialization=False)
                    tokenizer.save_pretrained(checkpoint_path) # 同时保存tokenizer
                    logger.info(f"检查点已保存到: {checkpoint_path}")

        if completed_steps >= args.max_train_steps:
            logger.info("已达到最大训练步数。")
            break

        if eval_dataloader:
            model.eval()

            # 用于累积整个评估周期的总损失和总指标值
            epoch_total_sum_loss = 0.0
            epoch_metric_sums = {}  # 例如: {"recon": 0.0, "comp": 0.0, ...}

            # 获取模型输出中除了'loss'以外的其他指标键 (只执行一次)
            # 假设在第一次迭代前，我们可以通过一个样本输出来确定这些键
            # 或者预先定义好你期望跟踪的指标键
            # 为简化，我们假设在第一次迭代时获取
            _temp_metric_keys = None

            epoch_total_processed_samples = 0

            logger.info(f"--- 开始在epoch {epoch + 1}结束时进行评估 ---")
            eval_progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process,
                                     desc="评估进度")

            with torch.no_grad():
                for batch_idx, eval_batch in enumerate(eval_dataloader):
                    batch_size_on_this_device = eval_batch["input_ids"].size(0)

                    outputs = model(
                        input_ids=eval_batch["input_ids"],
                        attention_mask=eval_batch["attention_mask"],
                        current_train_step=completed_steps  # 评估时退火参数通常固定
                    )

                    # 1. 处理总损失 (loss)
                    # outputs["loss"] 是当前设备上这个batch的平均损失
                    mean_loss_on_device = outputs["loss"].detach()
                    sum_loss_for_batch_on_device = mean_loss_on_device * batch_size_on_this_device

                    # 收集所有设备上的批次总损失
                    gathered_batch_sum_losses = accelerator.gather_for_metrics(
                        sum_loss_for_batch_on_device.reshape(1))  # Reshape to ensure it's a tensor
                    epoch_total_sum_loss += gathered_batch_sum_losses.sum().item()

                    # 2. 处理其他指标
                    if batch_idx == 0:  # 在第一个batch初始化metric_keys和epoch_metric_sums
                        _temp_metric_keys = [k for k in outputs.keys() if
                                             k != "loss" and isinstance(outputs[k], torch.Tensor)]
                        for key in _temp_metric_keys:
                            epoch_metric_sums[key] = 0.0

                    if _temp_metric_keys:  # 确保已初始化
                        for key in _temp_metric_keys:
                            if key in outputs and isinstance(outputs[key], torch.Tensor):
                                # outputs[key] 是当前设备上这个batch该指标的平均值
                                mean_metric_on_device = outputs[key].detach()
                                sum_metric_for_batch_on_device = mean_metric_on_device * batch_size_on_this_device

                                gathered_batch_sum_metrics = accelerator.gather_for_metrics(
                                    sum_metric_for_batch_on_device.reshape(1))
                                epoch_metric_sums[key] += gathered_batch_sum_metrics.sum().item()

                    # 3. 累积处理的样本总数
                    # (确保只加一次，即使有多个指标)
                    gathered_batch_sizes = accelerator.gather_for_metrics(
                        torch.tensor([batch_size_on_this_device], device=accelerator.device, dtype=torch.long)
                    )
                    epoch_total_processed_samples += gathered_batch_sizes.sum().item()

                    eval_progress_bar.update(1)
            eval_progress_bar.close()

            # 4. 计算整个评估周期的平均指标
            log_eval_metrics = {}
            if epoch_total_processed_samples > 0:
                avg_epoch_eval_loss = epoch_total_sum_loss / epoch_total_processed_samples
                log_eval_metrics["eval_loss"] = avg_epoch_eval_loss

                if _temp_metric_keys:
                    for key in _temp_metric_keys:
                        # 保持原始日志命名 eval_recon, eval_comp 等
                        log_key_name = f"eval_{key.replace('metrics_', 'metric_')}"
                        log_eval_metrics[log_key_name] = epoch_metric_sums[key] / epoch_total_processed_samples
            else:
                logger.warning("评估期间未处理任何样本。")
                log_eval_metrics["eval_loss"] = 0.0  # 或者 float('nan')
                if _temp_metric_keys:
                    for key in _temp_metric_keys:
                        log_key_name = f"eval_{key.replace('metrics_', 'metric_')}"
                        log_eval_metrics[log_key_name] = 0.0  # 或者 float('nan')

            accelerator.log(log_eval_metrics, step=completed_steps)
            logger.info(f"Epoch {epoch + 1} 评估结果: {json.dumps(log_eval_metrics, indent=2)}")


    progress_bar.close()
    logger.info("***** 训练完成 *****")

    # --- 13. 保存最终模型和tokenizer ---
    accelerator.wait_for_everyone() # 等待所有进程完成
    if accelerator.is_main_process:
        final_model_path = Path(args.output_dir) / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"保存最终模型到: {final_model_path}")
        unwrapped_model = accelerator.unwrap_model(model) # 获取原始模型
        unwrapped_model.save_pretrained(final_model_path) # 保存模型权重和配置文件(model_config.json)

        logger.info(f"保存最终tokenizer到: {final_model_path}")
        tokenizer.save_pretrained(final_model_path) # 在模型目录中也保存一份tokenizer

        # 保存训练参数和最终模型配置
        with open(final_model_path / "training_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        # model_config.json 应该已被 save_pretrained 保存，但可以再确认一下
        # with open(final_model_path / "model_config.json", "w") as f:
        #     json.dump(model_config.to_dict(), f, indent=2)

        logger.info("所有最终产物已保存。")

        # --- 14. (可选) 运行快速演示 ---
        # 这个演示需要适配MLM模型的特性，例如对输入进行mask然后预测被mask的词
        logger.info("运行快速MLM演示...")
        if len(train_dataset) > 0: # 从训练集中取一个样本
            sample_text_original = train_dataset[0][args.text_column][:model_config.max_seq_len - 20] # 截断以防超长

            # 准备输入，并手动mask一些token
            inputs = tokenizer(sample_text_original, return_tensors="pt", truncation=True, max_length=model_config.max_seq_len)
            input_ids_for_demo = inputs.input_ids.clone().to(accelerator.device) # 确保在正确设备上

            # 随机选择一些token进行mask (非特殊token)
            num_tokens_to_mask = max(1, int(input_ids_for_demo.size(1) * 0.15)) # mask 15%
            # 获取非特殊token的索引
            non_special_indices = (input_ids_for_demo[0] != tokenizer.cls_token_id) & \
                                  (input_ids_for_demo[0] != tokenizer.sep_token_id) & \
                                  (input_ids_for_demo[0] != tokenizer.pad_token_id) & \
                                  (input_ids_for_demo[0] != tokenizer.mask_token_id) # 避免重复mask

            candidate_indices_to_mask = non_special_indices.nonzero(as_tuple=False).squeeze()
            if candidate_indices_to_mask.numel() > num_tokens_to_mask:
                perm = torch.randperm(candidate_indices_to_mask.size(0), device=accelerator.device)
                indices_to_mask = candidate_indices_to_mask[perm[:num_tokens_to_mask]]
                input_ids_for_demo[0, indices_to_mask] = tokenizer.mask_token_id
                masked_text_input = tokenizer.decode(input_ids_for_demo[0], skip_special_tokens=False) # 保留特殊token看效果
            else: # 如果没有足够token可mask，就不mask
                indices_to_mask = None
                masked_text_input = sample_text_original
                logger.warning("演示样本中没有足够的可mask token。")

            unwrapped_model.eval() # 设置为评估模式
            with torch.no_grad():
                # 模型forward需要嵌入，而不是ids
                # 注意：MERGELanguageModel的forward需要current_train_step，评估时可以设为max_steps
                demo_attention_mask = (input_ids_for_demo != tokenizer.pad_token_id).long()

                # 直接调用MLM Encoder部分进行预测 (如果只想看MLM能力)
                # 1. 获取嵌入
                demo_word_embeds = unwrapped_model.word_embeddings(input_ids_for_demo)
                demo_input_embeds = unwrapped_model._apply_positional_embeddings(demo_word_embeds)
                # 2. 通过MLM Encoder
                mlm_output = unwrapped_model.mlm_encoder(
                    input_embeds=demo_input_embeds,
                    attention_mask=demo_attention_mask
                )
                predicted_logits = mlm_output["logits"] # (1, SeqLen, VocabSize)
                predicted_token_ids = torch.argmax(predicted_logits, dim=-1) # (1, SeqLen)

            # 将预测的token填回被mask的位置
            filled_ids = input_ids_for_demo.clone()
            if indices_to_mask is not None and indices_to_mask.numel() > 0:
                filled_ids[0, indices_to_mask] = predicted_token_ids[0, indices_to_mask]

            reconstructed_text_demo = tokenizer.decode(filled_ids[0], skip_special_tokens=True) # 跳过特殊token看纯文本

            demo_output_data = {
                "original_input": sample_text_original,
                "masked_input_text (with special tokens)": masked_text_input,
                "reconstructed_output_text (MLM filled)": reconstructed_text_demo,
            }
            if indices_to_mask is not None and indices_to_mask.numel() > 0:
                 demo_output_data["original_tokens_at_masked_positions"] = tokenizer.decode(inputs.input_ids[0, indices_to_mask].tolist())
                 demo_output_data["predicted_tokens_at_masked_positions"] = tokenizer.decode(predicted_token_ids[0, indices_to_mask].tolist())


            demo_output_file = Path(args.output_dir) / "quick_demo_output.json"
            with open(demo_output_file, "w", encoding="utf-8") as f:
                json.dump(demo_output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"MLM演示结果已保存到: {demo_output_file}")
        else:
            logger.info("训练集为空，跳过快速演示。")

    accelerator.end_training() # 清理tracker等资源
    logger.info("脚本执行完毕。")


if __name__ == "__main__":
    main()


