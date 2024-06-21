# Training script to fine-tune a pre-train LLM with QLoRA using HuggingFace.

import os
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
import evaluate
import numpy as np

import torch
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, CustomCallback

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)

def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with QLoRA")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Checkpoints to path of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store the fine-tuned model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--set_pad_id",
        action="store_true",
        help="Set the id for the padding token, needed by models such as Mistral-7B",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Eval batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="Lora rank"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=64, help="Lora alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="Lora dropout"
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default='none',
        choices={"lora_only", "none", 'all'},
        help="Layers to add learnable bias"
    )

    arguments = parser.parse_args()
    return arguments

def get_lora_model(model_checkpoints, rank=4, alpha=16, lora_dropout=0.1, bias='none'):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_checkpoints,
        device_map="auto",
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if model_checkpoints == 'mistralai/Mistral-7B-v0.1' or model_checkpoints == 'meta-llama/Llama-2-7b-hf':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
        )

    return model, tokenizer, peft_config


def get_unlearn_dataset_and_collator(
        data_path,
        tokenizer,
        add_prefix_space=True,
        max_length=1024,
        truncation=True
):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    # Tokenize inputs
    def _preprocessing_sentiment(examples):
        return {"text": prompt_template(examples['text'], examples['label_text'])}

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data = load_dataset(data_path)

    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(['label', 'label_text'])
    data.set_format("torch")

    print(data)

    return data, data_collator


def main(args):
    if 'llama-2-7b' in args.model_checkpoints.lower():
        model_name = 'llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_checkpoints.lower():
        model_name = 'llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_checkpoints.lower():
        model_name = 'opt-1.3b'
    else:
        raise NotImplementedError
    
    # Sync to wandb
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    os.environ["WANDB_PROJECT"] = f'qlora_{model_name.lower()}_{args.dataset.lower()}'  # log to your project
    
    data_path = get_data_path(args.dataset)

    if args.output_path is None:
        args.output_path = f'qlora_checkpoints/{model_name.lower()}-hf-qlora-{args.dataset.lower()}'

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                f.write(f'{k}: {v}\n')

    # Initialize models and collator
    model, tokenizer, lora_config = get_lora_model(
        args.model_checkpoints,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        add_prefix_space=True,
        truncation=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="no",
        save_strategy="no",
        group_by_length=True,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'lr={args.lr}',
        max_grad_norm=0.3,
        metric_for_best_model="eval_test_loss",
    )

    summary(model)

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field='text',
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
        train_dataset=concatenate_datasets([dataset['train_retain'], dataset['train_forget']]),
        eval_dataset={"test": concatenate_datasets([dataset['test_retain'], dataset['test_forget']])},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomCallback(trainer))
    start = time.perf_counter()
    trainer.train()
    runtime = (time.perf_counter()-start)
    print(runtime)


if __name__ == "__main__":
    args = get_args()
    main(args)
