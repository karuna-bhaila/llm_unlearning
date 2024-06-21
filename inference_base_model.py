# Script to run inference on bse model fine-tuned with QLoRA

import os
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser

import torch
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, LoraConfig, TaskType, PromptEncoderConfig, PeftConfig, PeftModel
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, \
    BitsAndBytesConfig
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, CustomCallback

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)


def get_args():
    parser = ArgumentParser(description="Run inference on base model fine-tuned with QLoRA")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--model_checkpoints",
        type=str,
        default=None,
        required=True,
        help="Checkpoints to path of the fine-tuned LLM",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to output folder",
    )

    arguments = parser.parse_args()
    return arguments


def get_lora_model(model_checkpoints, max_length, truc):
    lora_peft_model_id = model_checkpoints
    lora_config = PeftConfig.from_pretrained(lora_peft_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(lora_config.base_model_name_or_path,
                                                      device_map="auto",
                                                      offload_folder="offload",
                                                      trust_remote_code=True, )
    
    tokenizer = AutoTokenizer.from_pretrained(
        lora_config.base_model_name_or_path, 
        truncation=True, 
        padding=True, 
        max_length=max_length
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = PeftModel.from_pretrained(base_model, lora_peft_model_id)
    model = model.merge_and_unload()

    for n, p in model.named_parameters():
        if p.requires_grad:
            if "score" in n:
                print(f"Turning {n} to untrainable")
                p.requires_grad = False
            else:
                print(f"{n} is trainable")
    summary(model)

    return model, tokenizer, lora_config


def get_unlearn_dataset_and_collator(
        data_path,
        model_checkpoints,
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

    # Sync to wandb
    os.environ["WANDB_LOG_MODEL"] = "all"  
    os.environ["WANDB_PROJECT"] = f'inference_qlora_{model_name.lower()}_{args.dataset.lower()}' 

    data_path = get_data_path(args.dataset)

    # Load arguments of saved model
    path = os.path.dirname(args.model_checkpoints)
    with open(os.path.join(path, 'arguments.txt'), 'r') as f:
        parameters = f.readlines()
        params = {}
        for line in parameters:
            k, v = line.strip().split(':')
            params[k.strip()] = v.strip()

    # Initialize models
    model, tokenizer, lora_config = get_lora_model(
        args.model_checkpoints,
        max_length=int(params['max_length'])
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        args.dataset.lower(),
        args.model_name,
        tokenizer=tokenizer,
        max_length=int(params['max_length']),
        add_prefix_space=True,
        truncation=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=float(params['lr']),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=int(params['train_batch_size']),
        per_device_eval_batch_size=int(params['eval_batch_size']),
        num_train_epochs=int(params['num_epochs']),
        weight_decay=float(params['weight_decay']),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        group_by_length=True,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'{epoch}_lr={params["lr"]}',
        max_grad_norm=0.3,
    )

    if params['set_pad_id']=='True':
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field='text',
        max_seq_length=int(params['max_length']),
        tokenizer=tokenizer,
        train_dataset=concatenate_datasets([dataset['train_retain'], dataset['train_forget']]),
        eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget'],
                      "test_retain": dataset['test_retain'],
                      "test_forget": dataset['test_forget']},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomCallback(trainer))
    trainer.evaluate()


if __name__ == "__main__":
    args = get_args()
    main(args)
