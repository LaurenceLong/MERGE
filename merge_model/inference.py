import argparse
import logging

import torch
from transformers import AutoTokenizer
import Levenshtein # pip install python-Levenshtein

from .model import MERGELanguageModel # Assuming MERGE is in PYTHONPATH
from .configs import MERGEModelConfig # To access config defaults if needed
from .utils import calculate_mask_ratio # For t_infer

def calculate_t_infer(mask_ratio_S_hat: float) -> float:
    """
    Calculate pseudo-time condition for inference.
    Example: 1.0 - mask_ratio (more masks = earlier time = lower t_infer)
    Or could be more complex based on scheme's t_infer(mask_ratio(Ŝ)).
    """
    return 1.0 - mask_ratio_S_hat # Simple example

def main():
    parser = argparse.ArgumentParser(description="Inference with MERGE × MDLM Model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory.")
    parser.add_argument("--text", type=str, default="This is an example to test.")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length for processing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_len_heuristic", type=int, default=0, help="Optional target length for K_ins calc, 0 means use original text length.")


    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_config = MERGEModelConfig.from_pretrained(args.model_path) # Load config used for training
    model = MERGELanguageModel.from_pretrained(args.model_path, config=model_config) # Pass config
    model.to(args.device)
    model.eval()

    logger = logging.getLogger(__name__) # For info
    logger.setLevel(logging.INFO)

    # --- Prepare Initial Input (X) ---
    # Tokenize prompt or start with empty if generating from scratch
    if args.text.strip():
        # Bos token might be needed if model was trained with it on Y*
        input_text = f"{tokenizer.bos_token if tokenizer.bos_token else ''} {args.text.strip()} {tokenizer.eos_token if tokenizer.eos_token else ''}"
        tokenized_input = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True
        )
        X_tokens = tokenized_input.input_ids.to(args.device)
        X_attention_mask = tokenized_input.attention_mask.to(args.device)
    else: # Start empty (e.g. unconditional generation, though MERGE is more for editing)
        # Create a BOS token if available, then pad.
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else (
                   tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0) # Fallback
        X_tokens = torch.full((1, args.max_length), tokenizer.pad_token_id, dtype=torch.long, device=args.device)
        X_tokens[0, 0] = bos_id
        X_attention_mask = (X_tokens != tokenizer.pad_token_id).long()

    logger.info(f"Initial X: {tokenizer.decode(X_tokens[0][X_attention_mask[0].bool()], skip_special_tokens=False)}")

    prev_e_mdlm_output_for_gate = None # For first round
    
    # --- Iterative Editing/Generation Loop ---
    for l_round in range(model_config.max_rounds_l):
        logger.info(f"\n--- Round {l_round + 1} ---")
        
        # 1. Gate Decision
        tau_edit_l = 0.4 + 0.1 * l_round # As per scheme
        # Gate needs X_{l-1} (which is current X_tokens) for PPL calculation.
        g_decision = model.infer_gate(
            prev_e_mdlm_output_for_gate, 
            X_tokens, X_attention_mask, # X_tokens is X_{l-1} for Gate's PPL
            l_round, 
            tau_edit_l # tau_edit might be used as a threshold in infer_gate
        ) # Returns (B,) tensor of 0s or 1s. Here B=1.
        
        is_edit_path = g_decision.item() == 1
        logger.info(f"Gate decision: {'EDIT' if is_edit_path else 'GENERATE'}")

        # 2. Deletion (M₁) if Edit Path
        if is_edit_path:
            current_X_len_unpadded = X_attention_mask[0].sum().item()
            K_del = model._calculate_k_del(current_X_len_unpadded) # Use model's internal helper
            logger.info(f"Calculated K_del: {K_del}")
            if K_del > 0 :
                # infer_m1_delete_bottomk returns S_tokens (padded to X_tokens.shape[1])
                S_tokens_after_m1 = model.infer_m1_delete_bottomk(
                    X_tokens, X_attention_mask, K_del, model_config.m1_tau_del_end # Use final tau_del
                )
            else:
                S_tokens_after_m1 = X_tokens.clone()
            S_attention_mask_after_m1 = (S_tokens_after_m1 != tokenizer.pad_token_id).long()
        else: # Generate Path (no deletion)
            S_tokens_after_m1 = X_tokens.clone()
            S_attention_mask_after_m1 = X_attention_mask.clone()
        
        logger.info(f"S after M1: {tokenizer.decode(S_tokens_after_m1[0][S_attention_mask_after_m1[0].bool()], skip_special_tokens=False)}")

        # 3. Insertion (M₂)
        current_S_len_unpadded = S_attention_mask_after_m1[0].sum().item()
        # K_ins calc_k(|S|) = clamp(round(0.25·(|Y*|−|S|)), 1, 20)
        # For inference, |Y*| is not directly known. Heuristic: target_len or original_len.
        # Let's use args.max_length as a proxy for |Y*| or a user-defined target.
        target_y_star_len = args.target_len_heuristic if args.target_len_heuristic > 0 else X_attention_mask[0].sum().item() # original length of X
        target_y_star_len = max(target_y_star_len, current_S_len_unpadded) # Ensure non-negative diff

        K_ins = model._calculate_k_ins(current_S_len_unpadded, target_y_star_len)
        K_ins = min(K_ins, max(0, args.max_length - current_S_len_unpadded)) # Ensure not exceeding max_length
        logger.info(f"Calculated K_ins: {K_ins}")

        if K_ins > 0:
            # infer_m2_insert_masks returns counts_g (B, G_max)
            gap_insertion_counts = model.infer_m2_insert_masks(
                S_tokens_after_m1, S_attention_mask_after_m1, K_ins
            )
            # Perform actual insertion using these counts
            # _insert_masks_at_gaps(S_hard_tokens, S_hard_attention_mask, gap_insertion_counts, max_output_len)
            S_hat_hard_tokens, S_hat_hard_attention_mask = model._insert_masks_at_gaps(
                S_tokens_after_m1, S_attention_mask_after_m1, gap_insertion_counts, args.max_length
            )
        else:
            S_hat_hard_tokens = S_tokens_after_m1.clone() # No insertion
            S_hat_hard_attention_mask = S_attention_mask_after_m1.clone()

        logger.info(f"Ŝ (with masks): {tokenizer.decode(S_hat_hard_tokens[0][S_hat_hard_attention_mask[0].bool()], skip_special_tokens=False)}")

        # 4. Fuse & Fill (E_MDLM)
        # Simplified fuse for inference: just embed S_hat_hard_tokens
        S_hat_mixed_embeds = model.word_embeddings(S_hat_hard_tokens)
        
        mask_ratio_S_hat = calculate_mask_ratio(S_hat_hard_tokens, model_config.mask_token_id, S_hat_hard_attention_mask)
        t_infer_val = calculate_t_infer(mask_ratio_S_hat)
        pseudo_time_cond_tensor = torch.tensor([[t_infer_val]], device=args.device) # (1,1) for batch size 1

        e_mdlm_output = model.e_mdlm(
            input_embeds=S_hat_mixed_embeds,
            attention_mask=S_hat_hard_attention_mask,
            pseudo_time_condition=pseudo_time_cond_tensor
        )
        logits_current_round = e_mdlm_output["logits"]
        
        # Store EMDLM output for next round's Gate
        prev_e_mdlm_output_for_gate = {
            "logits": logits_current_round, # No detach needed for inference if not backpropping
            "hidden_states": e_mdlm_output["hidden_states"],
            "source_attention_mask": S_hat_hard_attention_mask # Mask of EMDLM's input
        }

        # Greedy fill MASKs and get confidence
        X_new_tokens, confidence = model.greedy_fill_with_confidence(S_hat_hard_tokens, logits_current_round)
        X_new_attention_mask = (X_new_tokens != tokenizer.pad_token_id).long()
        
        logger.info(f"X_new (filled): {tokenizer.decode(X_new_tokens[0][X_new_attention_mask[0].bool()], skip_special_tokens=True)}")
        logger.info(f"Confidence of fill: {confidence.item():.4f}")

        # 5. Stopping Conditions
        # Levenshtein distance between X_new (string) and X (string)
        # Need to decode X_tokens and X_new_tokens (unpadded, skip special)
        X_str_old = tokenizer.decode(X_tokens[0][X_attention_mask[0].bool()], skip_special_tokens=True)
        X_str_new = tokenizer.decode(X_new_tokens[0][X_new_attention_mask[0].bool()], skip_special_tokens=True)
        lev_dist = Levenshtein.distance(X_str_new, X_str_old)
        logger.info(f"Levenshtein distance to previous X: {lev_dist}")

        if confidence.item() > 0.9 or lev_dist < 2 or l_round == model_config.max_rounds_l - 1:
            logger.info("Stopping condition met.")
            X_tokens = X_new_tokens # Final output
            X_attention_mask = X_new_attention_mask
            break
        
        # Update X for next round
        X_tokens = X_new_tokens
        X_attention_mask = X_new_attention_mask

    # --- Final Output ---
    final_text_output = tokenizer.decode(X_tokens[0][X_attention_mask[0].bool()], skip_special_tokens=True)
    logger.info(f"\n--- Final Output Text ---")
    print(final_text_output)


if __name__ == "__main__":
    # Basic logging for the inference script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

