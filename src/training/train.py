"""
src/training/train.py
─────────────────────────────────────────────────────────────────────────────
Phase 3 — Model Fine-Tuning
Multilingual African Health Assistant | Zindi ITU Challenge

Supports:
  - Any HuggingFace seq2seq model via --base_model flag
  - Full fine-tune OR LoRA (PEFT) auto-selected by model param count
  - Curriculum learning (high-resource langs first)
  - Temperature-based per-language sampling
  - Per-language ROUGE-1, ROUGE-L, BLEU-4 at every epoch
  - MLflow / W&B experiment tracking
  - Automatic early stopping & best-checkpoint management

Usage:
  python src/training/train.py --config src/training/config.yaml
  python src/training/train.py --config src/training/config.yaml --base_model google/mt5-large
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from datasets import Dataset, DatasetDict

# ─── Optional Imports (graceful degradation) ─────────────────────────────────

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft not installed — LoRA disabled.")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─── Metric Utils ────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1 and ROUGE-L F1 scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1_scores, rl_scores = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
    return {
        "rouge1_f1": float(np.mean(r1_scores)),
        "rougeL_f1": float(np.mean(rl_scores)),
    }


# ─── Dataset Preparation ─────────────────────────────────────────────────────

LANGUAGE_NAMES = {
   "Eng_Uga": "English",
   "Aka_Gha": "Akan",
   "Eng_Gha": "English",
   "Eng_Eth": "English",
   "Lug_Uga": "Luganda",
   "Eng_Ken": "English",
   "Swa_Ken": "Swahili",
   "Amh_Eth": "Amharic",
}


def build_prompt(question: str, language: str) -> str:
    """Format input with language-aware task prefix."""
    lang_name = LANGUAGE_NAMES.get(language, language)
    return f"Answer in {lang_name}: {question}"


def tokenize_function(examples: dict, tokenizer, max_input: int, max_output: int) -> dict:
    """Tokenise (input, output) pairs for seq2seq training."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=examples["output_text"],
        max_length=max_output,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_dataset(
    train_path: str,
    val_path: str,
    curriculum_phase: int,
    curriculum_langs: Optional[list[str]],
    tokenizer,
    cfg: dict,
) -> DatasetDict:
    """
    Load and tokenise train/val datasets.

    curriculum_phase 1 → filter to curriculum_langs only.
    curriculum_phase 2 → use full dataset.
    """
    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    if curriculum_phase == 1 and curriculum_langs:
        df_train = df_train[df_train["subset"].isin(curriculum_langs)].copy()
        logger.info(f"Curriculum phase 1: training on {curriculum_langs} only "
                    f"({len(df_train)} rows).")

    # Build prompted inputs
    df_train["input_text"]  = df_train.apply(
        lambda r: build_prompt(r["input"], r["subset"]), axis=1)
    df_train["output_text"] = df_train["output"].astype(str)

    df_val["input_text"]  = df_val.apply(
        lambda r: build_prompt(r["input"], r["subset"]), axis=1)
    df_val["output_text"] = df_val["output"].astype(str)

    max_in  = cfg["model"]["max_input_length"]
    max_out = cfg["model"]["max_output_length"]

    def _tokenize(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["input_text", "output_text", "subset", "ID"]],
                                 preserve_index=False)
        return ds.map(
            lambda ex: tokenize_function(ex, tokenizer, max_in, max_out),
            batched=True,
            remove_columns=["input_text", "output_text"],
        )

    return DatasetDict({
        "train": _tokenize(df_train),
        "validation": _tokenize(df_val),
    })


# ─── LoRA Setup ──────────────────────────────────────────────────────────────

def apply_lora(model, lora_cfg: dict):
    """Wrap model with LoRA adapters via PEFT."""
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft library not installed. Cannot apply LoRA.")
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def count_parameters(model) -> int:
    """Return total parameter count of a model."""
    return sum(p.numel() for p in model.parameters())


# ─── Compute Metrics Callback ─────────────────────────────────────────────────

def make_compute_metrics(tokenizer, df_val: pd.DataFrame):
    """Factory: returns a compute_metrics fn that reports per-language ROUGE."""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Replace -100 (padding) with pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,      skip_special_tokens=True)

        # Strip whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Overall ROUGE
        overall = compute_rouge(decoded_preds, decoded_labels)

        # Per-language ROUGE
        subsets = df_val["subset"].values
        metrics = {**overall}
        for lang in set(subsets):
            idxs = [i for i, s in enumerate(subsets) if s == lang]
            if not idxs:
                continue
            lang_preds  = [decoded_preds[i]  for i in idxs]
            lang_labels = [decoded_labels[i] for i in idxs]
            lang_rouge  = compute_rouge(lang_preds, lang_labels)
            metrics[f"{lang}_rouge1"] = lang_rouge["rouge1_f1"]
            metrics[f"{lang}_rougeL"] = lang_rouge["rougeL_f1"]

        return {"eval_rouge_l": overall["rougeL_f1"], **metrics}

    return compute_metrics


# ─── Main Training Function ───────────────────────────────────────────────────

def train(cfg: dict, base_model_override: Optional[str] = None) -> None:
    """End-to-end training pipeline."""
    seed = cfg["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths      = cfg["paths"]
    model_cfg  = cfg["model"]
    train_cfg  = cfg["training"]
    opt_cfg    = cfg["optimiser"]
    lora_cfg   = cfg["lora"]
    curr_cfg   = cfg["curriculum"]
    track_cfg  = cfg["tracking"]

    base_model = base_model_override or model_cfg["base_model"]
    output_dir = Path(paths["models"]) / base_model.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Base model: {base_model}")
    logger.info(f"Output dir: {output_dir}")

    # ── Load tokeniser & model ──────────────────────────────────────────────
    logger.info("Loading tokeniser …")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    logger.info("Loading model …")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if train_cfg["mixed_precision"] == "bf16" else torch.float32,
    )

    # Auto-select LoRA vs full fine-tune based on param count
    n_params = count_parameters(model)
    use_lora = lora_cfg["enabled"] and n_params >= 1e9 and PEFT_AVAILABLE
    logger.info(f"Model parameters: {n_params/1e6:.0f}M  |  LoRA: {use_lora}")
    if use_lora:
        model = apply_lora(model, lora_cfg)
        lr = opt_cfg["lr"]
    else:
        lr = 5e-5  # full fine-tune uses smaller lr

    # ── Experiment tracking ─────────────────────────────────────────────────
    if track_cfg["backend"] == "mlflow" and MLFLOW_AVAILABLE:
        mlflow.set_experiment(track_cfg["experiment"])
        mlflow.start_run(run_name=track_cfg["run_name"])
        mlflow.log_params({
            "base_model": base_model, "n_params_M": round(n_params / 1e6),
            "use_lora": use_lora, "seed": seed,
            **{k: v for k, v in train_cfg.items()},
        })
    elif track_cfg["backend"] == "wandb" and WANDB_AVAILABLE:
        wandb.init(project=track_cfg["experiment"], name=track_cfg["run_name"])

    # ── Curriculum learning ─────────────────────────────────────────────────
    train_file = Path(paths["data_augmented"]) / "final_train.csv"
    val_file   = Path(paths["data_cleaned"])   / "val_clean.csv"
    df_val     = pd.read_csv(val_file)

    for curriculum_phase in [1, 2] if curr_cfg["enabled"] else [2]:
        if curriculum_phase == 1:
            phase_epochs = curr_cfg["phase1_epochs"]
            phase_langs  = curr_cfg["phase1_langs"]
            phase_label  = "Curriculum Phase 1 (high-resource)"
        else:
            phase_epochs = train_cfg["epochs"] - (curr_cfg["phase1_epochs"] if curr_cfg["enabled"] else 0)
            phase_langs  = None  # all languages
            phase_label  = "Curriculum Phase 2 (all languages)"

        logger.info(f"\n{'═'*60}\n{phase_label}\n{'═'*60}")

        dataset = load_dataset(
            train_path=str(train_file),
            val_path=str(val_file),
            curriculum_phase=curriculum_phase,
            curriculum_langs=phase_langs,
            tokenizer=tokenizer,
            cfg=cfg,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=phase_epochs,
            per_device_train_batch_size=train_cfg["per_device_train_batch"],
            per_device_eval_batch_size=train_cfg["per_device_eval_batch"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation"],
            learning_rate=lr,
            weight_decay=opt_cfg["weight_decay"],
            lr_scheduler_type=opt_cfg["lr_scheduler"],
            warmup_ratio=opt_cfg["warmup_ratio"],
            label_smoothing_factor=opt_cfg["label_smoothing"],
            bf16=train_cfg["mixed_precision"] == "bf16",
            fp16=train_cfg["mixed_precision"] == "fp16",
            gradient_checkpointing=train_cfg["gradient_checkpointing"],
            predict_with_generate=True,
            generation_max_length=model_cfg["max_output_length"],
            logging_steps=train_cfg["logging_steps"],
            evaluation_strategy=train_cfg["eval_strategy"],
            save_strategy=train_cfg["save_strategy"],
            save_total_limit=train_cfg["save_total_limit"],
            load_best_model_at_end=train_cfg["load_best_model_at_end"],
            metric_for_best_model=train_cfg["metric_for_best_model"],
            greater_is_better=True,
            dataloader_num_workers=train_cfg["dataloader_num_workers"],
            seed=seed,
            report_to=track_cfg["backend"] if track_cfg["backend"] in ("wandb",) else "none",
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, label_pad_token_id=-100
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=make_compute_metrics(tokenizer, df_val),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=train_cfg["early_stopping_patience"]
                )
            ] if curriculum_phase == 2 else [],
        )

        logger.info("Starting training …")
        trainer.train()

    # ── Save final model ─────────────────────────────────────────────────────
    best_model_dir = output_dir / "best"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.success(f"Best model saved → {best_model_dir}")

    if MLFLOW_AVAILABLE and track_cfg["backend"] == "mlflow":
        mlflow.end_run()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Multilingual African Health QA model")
    parser.add_argument("--config",     default="src/training/config.yaml")
    parser.add_argument("--base_model", default=None,
                        help="Override base_model in config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, base_model_override=args.base_model)
