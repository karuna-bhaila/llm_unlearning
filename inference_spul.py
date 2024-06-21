# Script to run inference using SPUL

import os
import pickle
import datetime
import random
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, PeftConfig, PeftModel
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from trl import DataCollatorForCompletionOnlyLM

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, get_logits_from_base_model, CustomCallback
from spul import get_unlearn_dataset_and_collator, get_unlearning_loss_trainer

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)


def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with PEFT")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        default=None,
        required=True,
        help="Name of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--logits_path",
        type=str,
        default=None,
        required=False,
        help="Path to save original logits to use for KL loss",
    )
    parser.add_argument(
        "--forget_size",
        type=float,
        default=1.0,
        required=False,
        help="relative size of forget set for ablation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store the fine-tuned model",
    )

    arguments = parser.parse_args()
    return arguments


def get_ptuning_model(model_checkpoints, lora_checkpoints, max_length):
    lora_peft_model_id = lora_checkpoints
    ptuning_model_id = model_checkpoints

    lora_config = PeftConfig.from_pretrained(lora_peft_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(lora_config.base_model_name_or_path,
                                                      device_map="auto",
                                                      offload_folder="offload",
                                                      trust_remote_code=True, )
    
    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path, truncation=True, padding=True,
                                              max_length=max_length)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    lora_model = PeftModel.from_pretrained(base_model, lora_peft_model_id)
    lora_model = lora_model.merge_and_unload()
    
    model = PeftModel.from_pretrained(lora_model, ptuning_model_id)

    for n, p in model.named_parameters():
        if p.requires_grad:
            if "score" in n:
                print(f"Turning {n} to untrainable")
                p.requires_grad = False
            else:
                print(f"{n} is trainable")
    summary(model)
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def main(args):
    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  

    if 'llama-2-7b' in args.model_checkpoints.lower():
        model_name = 'llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_checkpoints.lower():
        model_name = 'llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_checkpoints.lower():
        model_name = 'opt-1.3b'
    
    os.environ["WANDB_PROJECT"] = f'spul_inference_{model_name}_{args.dataset.lower()}'  

    data_path = get_data_path(args.dataset)

    if args.logits_path is None:
        args.logits_path = f'saved_logits/{model_name}_{args.dataset.lower()}-{args.forget_size}.pkl'

    # Load arguments from saved model
    path = os.path.dirname(args.model_checkpoints)
    with open(os.path.join(path, 'arguments.txt'), 'r') as f:
        parameters = f.readlines()
    params = {}
    for line in parameters:
        k, v = line.strip().split(':')
        params[k.strip()] = v.strip()

    model, tokenizer = get_ptuning_model(
        args.model_checkpoints,
        params['model_name'],
        int(params['max_length'])
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        data_path,
        args.model_checkpoints,
        tokenizer=tokenizer,
        max_length=int(params['max_length']),
        add_prefix_space=True,
        truncation=True,
    )    

    with open(args.logits_path, 'rb') as f:
        original_logits = pickle.load(f)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=float(params['lr']),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=int(params['train_batch_size']),
        per_device_eval_batch_size=int(params['eval_batch_size']),
        num_train_epochs=int(params['eval_batch_size']),
        weight_decay=int(params['eval_batch_size']),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        group_by_length=True,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'lr={params["lr"]}_alpha={params["alpha"]}_beta={params["beta"]}_numtokens={params["ptuning_num_tokens"]}',
        max_grad_norm=0.3,
        remove_unused_columns=False,
    )

    if params['set_pad_id'] == 'True':
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    custom_loss = get_unlearning_loss_trainer()

    trainer = custom_loss(
        model=model,
        original_logits=original_logits,
        num_virtual_tokens=int(params['ptuning_num_tokens']),
        alpha=float(params['alpha']),
        beta=float(params['beta']),
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget'],
                      "test_retain": dataset['test_retain'],
                      "test_forget": dataset['test_forget']},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomCallback(trainer))
    results = trainer.evaluate()
    # print(results)


if __name__ == "__main__":
    args = get_args()
    main(args)