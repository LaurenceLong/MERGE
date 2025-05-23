import argparse
import json
import logging
from pathlib import Path
import math

import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer # Added AutoTokenizer

from .configs import MERGEModelConfig
from .data import build_custom_tokenizer, collate_fn_for_merge # Assuming collate_fn is okay
from .model import MERGELanguageModel
from .utils import linear_anneal # For annealing taus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def parse_training_arguments():
    parser = argparse.ArgumentParser(description="Train MERGE × MDLM Model.")
    # Add relevant arguments from MERGEModelConfig and new scheme
    parser.add_argument("--dataset_path", type=str, default="wikitext")
    parser.add_argument("--dataset_name", type=str, default="wikitext-2-v1")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--validation_split_percentage", type=int, default=5)
    
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=50000) # Reduced for quicker test if needed
    parser.add_argument("--batch_size_train_per_device", type=int, default=16) # Reduced due to iterative nature
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1) # Adjusted
    
    parser.add_argument("--output_dir", type=str, default="outputs_merge_mdlm")
    parser.add_argument("--tokenizer_save_dir", type=str, default=None)
    parser.add_argument("--log_every_steps", type=int, default=50) # Log more frequently
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--eval_every_steps", type=int, default=500) # Evaluate periodically
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard_log_name", type=str, default="merge_mdlm_logs")

    # Freezing strategy related (simplified)
    parser.add_argument("--train_e_mdlm_only_epochs", type=int, default=0, help="Epochs to train E_MDLM only (Gate fixed to edit, M1/M2 active but maybe gentle).")
    parser.add_argument("--train_m1_m2_e_mdlm_epochs", type=int, default=0, help="Epochs to train M1/M2/E_MDLM (Gate fixed or learns).")
    # Full training happens after these.

    args = parser.parse_args()
    if args.tokenizer_save_dir is None:
        args.tokenizer_save_dir = str(Path(args.output_dir) / "tokenizer")
    return args

def main():
    args = parse_training_arguments()

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=Path(args.output_dir) / args.tensorboard_log_name
    )
    accelerator = Accelerator(
        log_with="tensorboard",
        project_config=accelerator_project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if torch.cuda.is_available() else "no" # Check for cuda for fp16
    )

    if accelerator.is_local_main_process:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
        logger.setLevel(logging.WARNING)

    logger.info(f"Accelerator state: {accelerator.state}")
    if args.seed is not None:
        set_seed(args.seed)

    # --- Model Config ---
    config_overrides = {k: v for k, v in vars(args).items() if v is not None and hasattr(MERGEModelConfig, k)}
    model_config = MERGEModelConfig(**config_overrides)
    # Update max_seq_len if provided by args
    if args.max_seq_len is not None: model_config.max_seq_len = args.max_seq_len
    logger.info(f"Model Config: {model_config.to_json_string()}")

    # --- Tokenizer ---
    # tokenizer = build_custom_tokenizer(...) # Your existing function
    # For MERGE × MDLM, ensure PAD, MASK, BOS, EOS are set.
    # Using AutoTokenizer for simplicity here, assuming a pretrained one like 'gpt2' or a path.
    if Path(args.tokenizer_save_dir).exists():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_save_dir, trust_remote_code=True)
        logger.info(f"Loaded tokenizer from {args.tokenizer_save_dir}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
            logger.info(f"Set PAD token to {tokenizer.pad_token}")
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            logger.info("Added MASK token: [MASK]")
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({'bos_token': '[BOS]'})
            logger.info("Added BOS token: [BOS]")
        if tokenizer.eos_token is None: # Should exist if pad_token was set to it
             tokenizer.add_special_tokens({'eos_token': '[EOS]'})
             logger.info("Added EOS token: [EOS]")
        
        tokenizer.save_pretrained(args.tokenizer_save_dir)
        logger.info(f"Saved new/updated tokenizer to {args.tokenizer_save_dir}")

    model_config.vocab_size = len(tokenizer)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.mask_token_id = tokenizer.mask_token_id
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.eos_token_id
    
    if model_config.pad_token_id is None or model_config.mask_token_id is None:
        raise ValueError("PAD and MASK tokens must be set in tokenizer and config.")

    # --- Datasets & DataLoaders ---
    raw_datasets = load_dataset(args.dataset_path, args.dataset_name, trust_remote_code=True)
    # ... (train/validation split logic from your original script) ...
    if "validation" not in raw_datasets.keys() and args.validation_split_percentage > 0:
        train_validation_split = raw_datasets["train"].train_test_split(test_size=args.validation_split_percentage / 100.0, seed=args.seed)
        raw_datasets["train"] = train_validation_split["train"]
        raw_datasets["validation"] = train_validation_split["test"]
    elif "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = None

    def preprocess_function(examples):
        # Tokenize and ensure BOS/EOS if model expects it for Y*
        # For MERGE, Y* is the target. X_l starts as prompt or Y*.
        # Let's assume Y* is the full text.
        texts = [f"{tokenizer.bos_token} {text} {tokenizer.eos_token}" for text in examples[args.text_column]]
        tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=model_config.max_seq_len)
        # We need 'input_ids' (for Y*) and 'attention_mask'
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, num_proc=4,
            remove_columns=raw_datasets["train"].column_names
        )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"] if raw_datasets["validation"] else None

    # Collate function is simpler now if data is pre-tokenized
    def simple_collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=simple_collate_fn,
        batch_size=args.batch_size_train_per_device, num_workers=4
    )
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=simple_collate_fn,
            batch_size=args.batch_size_train_per_device * 2, num_workers=4
        )
    else:
        eval_dataloader = None
    
    # --- Model ---
    model = MERGELanguageModel(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_warmup_steps = int(args.max_train_steps * args.warmup_steps_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=args.max_train_steps
    )

    # --- Accelerator Prepare ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if eval_dataloader:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    # --- Tensorboard Init ---
    if accelerator.is_main_process:
        tb_tracker = accelerator.get_tracker("tensorboard")
        if tb_tracker:
            # tb_tracker.log_hyperparams(vars(args)) # This might not exist directly
            # tb_tracker.writer.add_text("hyperparameters", json.dumps(vars(args), indent=2))
            # tb_tracker.writer.add_text("model_config", model_config.to_json_string(use_diff=False))
            # Accelerator's log method is preferred for metrics. For text, use writer.
             pass # Will log metrics via accelerator.log

    # --- Training Loop ---
    logger.info("***** MERGE × MDLM Training *****")
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process, desc="Training")
    completed_steps = 0
    
    # Calculate anneal steps based on total training steps if not specified in config
    # Assuming anneal_steps in config are epochs, convert to steps
    # For simplicity, let's use a fraction of max_train_steps for annealing.
    # Example: anneal over first 25% of training.
    anneal_duration_steps = args.max_train_steps // 4 
    model_config.gate_tau_anneal_steps = anneal_duration_steps
    model_config.m1_tau_del_anneal_steps = anneal_duration_steps
    
    # Teacher forcing schedule
    # Convert epochs to steps
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    teacher_forcing_warmup_steps = model_config.teacher_forcing_warmup_epochs * steps_per_epoch
    teacher_forcing_decay_start_step = model_config.teacher_forcing_prob_decay_start_epoch * steps_per_epoch
    teacher_forcing_decay_end_step = model_config.teacher_forcing_prob_end_epoch * steps_per_epoch
    teacher_forcing_decay_duration = teacher_forcing_decay_end_step - teacher_forcing_decay_start_step
    if teacher_forcing_decay_duration <=0 : teacher_forcing_decay_duration = 1 # Avoid div by zero


    # Determine number of epochs based on max_steps
    num_epochs = math.ceil(args.max_train_steps / len(train_dataloader))
    logger.info(f"Total epochs (approx based on max_steps): {num_epochs}")


    for epoch in range(num_epochs):
        model.train()
        # TODO: Implement freezing strategy based on epoch and args.train_e_mdlm_only_epochs etc.
        # Example:
        # if epoch < args.train_e_mdlm_only_epochs:
        #   for name, param in model.named_parameters():
        #     if not name.startswith("e_mdlm."): param.requires_grad = False
        # elif epoch < args.train_e_mdlm_only_epochs + args.train_m1_m2_e_mdlm_epochs:
        #   for name, param in model.named_parameters():
        #     if name.startswith("gate_"): param.requires_grad = False # Or specific gate modules
        # else: # Full train
        #   for param in model.parameters(): param.requires_grad = True

        for step, batch in enumerate(train_dataloader):
            if completed_steps >= args.max_train_steps: break

            # Initialize for the sequence of L_max rounds
            # Y_star is the ground truth for the whole sequence
            Y_star_tokens_batch = batch["input_ids"] 
            Y_star_attention_mask_batch = batch["attention_mask"]

            # X_l starts as Y_star (or prompt in general, but Y_star for this training setup)
            X_l_tokens_current_iter = Y_star_tokens_batch.clone()
            X_l_attention_mask_current_iter = Y_star_attention_mask_batch.clone()
            
            prev_e_mdlm_output_for_gate = None # For the first round (l=0)
            
            total_loss_over_rounds = torch.tensor(0.0, device=accelerator.device)
            
            # --- Inner loop for L_max rounds ---
            for l_round in range(model_config.max_rounds_l):
                with accelerator.accumulate(model): # Accumulate gradients per round's forward pass
                    # Anneal temperatures for Gate and M1
                    current_tau_gate = linear_anneal(
                        completed_steps, model_config.gate_tau_start, model_config.gate_tau_end,
                        model_config.gate_tau_anneal_steps 
                    )
                    current_tau_del = linear_anneal(
                        completed_steps, model_config.m1_tau_del_start, model_config.m1_tau_del_end,
                        model_config.m1_tau_del_anneal_steps
                    )

                    # Model forward for one round
                    # The model's forward pass needs X_l, prev EMDLM output (for Gate), round_idx, Y_star (for L_recon), taus
                    model_outputs = model(
                        X_l_tokens=X_l_tokens_current_iter,
                        X_l_attention_mask=X_l_attention_mask_current_iter,
                        prev_e_mdlm_output=prev_e_mdlm_output_for_gate,
                        round_idx=l_round,
                        Y_star_tokens=Y_star_tokens_batch, # Ground truth
                        current_tau_gate=current_tau_gate,
                        current_tau_del=current_tau_del
                    )
                    
                    round_loss = model_outputs["loss"]
                    total_loss_over_rounds += round_loss # Accumulate loss over L_max rounds

                    # Prepare for next iteration (X_l and prev_e_mdlm_output)
                    # Store EMDLM output of this round for Gate of next round
                    prev_e_mdlm_output_for_gate = {
                        "logits": model_outputs["logits"].detach(), # Detach to prevent re-computation of grads
                        "hidden_states": model_outputs["hidden_states"].detach(),
                        "source_attention_mask": model_outputs["S_hat_hard_attention_mask"].detach()
                    }

                    # Teacher Forcing for next X_l
                    use_teacher_forcing = False
                    if completed_steps < teacher_forcing_warmup_steps:
                        use_teacher_forcing = True
                    elif completed_steps < teacher_forcing_decay_end_step : # In decay period
                        # Linear decay for teacher forcing probability
                        decay_progress = (completed_steps - teacher_forcing_decay_start_step) / teacher_forcing_decay_duration
                        current_tf_prob = model_config.teacher_forcing_initial_prob - \
                                          decay_progress * (model_config.teacher_forcing_initial_prob - model_config.teacher_forcing_final_prob)
                        current_tf_prob = max(model_config.teacher_forcing_final_prob, current_tf_prob)
                        if torch.rand(1).item() < current_tf_prob:
                            use_teacher_forcing = True
                    
                    if use_teacher_forcing:
                        X_l_tokens_current_iter = Y_star_tokens_batch.clone()
                        # X_l_attention_mask_current_iter remains Y_star_attention_mask_batch
                    else:
                        # Greedy fill from model's own predictions
                        # Ensure model instance is available (unwrap if needed for direct method call)
                        unwrapped_model = accelerator.unwrap_model(model)
                        X_l_tokens_current_iter = unwrapped_model.greedy_fill(
                            model_outputs["S_hat_hard_tokens"], model_outputs["logits"]
                        )
                    X_l_attention_mask_current_iter = (X_l_tokens_current_iter != model_config.pad_token_id).long()

                    # Break if X_l becomes stable or max rounds reached (handled by loop)
                    # This break logic is more for inference. In training, we run all L_max rounds.

            # --- End of L_max rounds ---
            # Average loss over rounds for backward pass
            avg_loss_for_sequence = total_loss_over_rounds / model_config.max_rounds_l
            
            accelerator.backward(avg_loss_for_sequence)
            if accelerator.sync_gradients: # After gradient accumulation
                if model_config.gradient_clip_val > 0:
                    accelerator.clip_grad_norm_(model.parameters(), model_config.gradient_clip_val)
            
            optimizer.step()
            lr_scheduler.step() # Step scheduler per optimizer step
            optimizer.zero_grad()

            # Update progress after processing one batch (all its rounds and grad accumulation)
            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)

                if completed_steps % args.log_every_steps == 0:
                    log_metrics = {
                        "train_loss_avg_rounds": avg_loss_for_sequence.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "completed_steps": completed_steps,
                        "tau_gate": current_tau_gate,
                        "tau_del": current_tau_del,
                        # Log other detached metrics from model_outputs if needed (e.g., last round's sub-losses)
                        "last_round_loss_recon": model_outputs["loss_reconstruction"].item(),
                        "last_round_loss_gate": model_outputs["loss_gate"].item(),
                        "last_round_loss_m1": model_outputs["loss_m1_entropy"].item(),
                        "last_round_loss_m2": model_outputs["loss_m2_entropy"].item(),
                    }
                    accelerator.log(log_metrics, step=completed_steps)
                    progress_bar.set_postfix(train_loss=f"{log_metrics['train_loss_avg_rounds']:.4f}")

                if completed_steps % args.save_every_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_path = Path(args.output_dir) / f"checkpoint_step_{completed_steps}"
                        accelerator.unwrap_model(model).save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"Checkpoint saved to {save_path}")
                
                if eval_dataloader and completed_steps % args.eval_every_steps == 0:
                    evaluate_model(model, eval_dataloader, accelerator, model_config, completed_steps, epoch)


        if completed_steps >= args.max_train_steps:
            logger.info("Max training steps reached.")
            break
    
    progress_bar.close()
    logger.info("***** Training Finished *****")

    # --- Save Final Model ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_path = Path(args.output_dir) / "final_model"
        accelerator.unwrap_model(model).save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        # Save training args
        with open(final_save_path / "training_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Final model saved to {final_save_path}")
    
    accelerator.end_training()


def evaluate_model(model, eval_dataloader, accelerator, model_config, completed_steps, epoch):
    model.eval()
    total_eval_loss_avg_rounds = 0
    eval_metrics_accum = {} # For other metrics from model output

    logger.info(f"--- Running evaluation at step {completed_steps} (Epoch {epoch+1}) ---")
    eval_progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, desc="Evaluating")

    with torch.no_grad():
        for batch in eval_dataloader:
            Y_star_tokens_batch = batch["input_ids"]
            X_l_tokens_current_iter = Y_star_tokens_batch.clone()
            X_l_attention_mask_current_iter = (X_l_tokens_current_iter != model_config.pad_token_id).long()
            prev_e_mdlm_output_for_gate = None
            current_eval_loss_over_rounds = torch.tensor(0.0, device=accelerator.device)
            
            last_round_outputs_for_log = None

            for l_round in range(model_config.max_rounds_l):
                # Use final (end) tau values for evaluation
                eval_tau_gate = model_config.gate_tau_end
                eval_tau_del = model_config.m1_tau_del_end

                model_outputs = model(
                    X_l_tokens=X_l_tokens_current_iter,
                    X_l_attention_mask=X_l_attention_mask_current_iter,
                    prev_e_mdlm_output=prev_e_mdlm_output_for_gate,
                    round_idx=l_round,
                    Y_star_tokens=Y_star_tokens_batch,
                    current_tau_gate=eval_tau_gate,
                    current_tau_del=eval_tau_del
                )
                current_eval_loss_over_rounds += model_outputs["loss"]
                prev_e_mdlm_output_for_gate = {
                    "logits": model_outputs["logits"], "hidden_states": model_outputs["hidden_states"],
                    "source_attention_mask": model_outputs["S_hat_hard_attention_mask"]
                }
                # For eval, X_l for next round is from greedy fill
                unwrapped_model = accelerator.unwrap_model(model) # Ensure direct method call
                X_l_tokens_current_iter = unwrapped_model.greedy_fill(
                    model_outputs["S_hat_hard_tokens"], model_outputs["logits"]
                )
                X_l_attention_mask_current_iter = (X_l_tokens_current_iter != model_config.pad_token_id).long()
                if l_round == model_config.max_rounds_l - 1:
                    last_round_outputs_for_log = model_outputs


            avg_eval_loss_for_seq = current_eval_loss_over_rounds / model_config.max_rounds_l
            total_eval_loss_avg_rounds += avg_eval_loss_for_seq.item()
            
            # Accumulate other metrics from the last round if needed
            if last_round_outputs_for_log:
                for k, v in last_round_outputs_for_log.items():
                    if "loss_" in k and isinstance(v, torch.Tensor): # e.g. loss_reconstruction
                        metric_name = f"eval_{k}"
                        eval_metrics_accum[metric_name] = eval_metrics_accum.get(metric_name, 0.0) + v.item()
            eval_progress_bar.update(1)
    
    eval_progress_bar.close()
    avg_total_eval_loss = total_eval_loss_avg_rounds / len(eval_dataloader)
    log_eval_metrics = {"eval_loss_avg_rounds": avg_total_eval_loss}
    
    for name, val in eval_metrics_accum.items():
        log_eval_metrics[name] = val / len(eval_dataloader)

    accelerator.log(log_eval_metrics, step=completed_steps)
    logger.info(f"Evaluation results at step {completed_steps}: {json.dumps(log_eval_metrics, indent=2)}")
    model.train() # Set back to train mode

if __name__ == "__main__":
    main()
