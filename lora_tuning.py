r"""
Training script to fine-tune a pre-train LLM with PEFT methods using HuggingFace.
  Example to run this conversion script:
    python lora_tuning.py \
     --dataset <path_to_dataset_folder> \
     --model_name <path_to_pre-trained_model> \
     --output_path <path_to_output_folder> \
"""

import os
import sys
import time

from torchinfo import summary
from copy import deepcopy

from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets
import evaluate
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainerState, TrainerControl, AutoModelForCausalLM, \
    BitsAndBytesConfig
from transformers import TrainingArguments, TrainerCallback
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)


def get_args():
    parser = ArgumentParser(description="Fine-tune an LLM model with PEFT")
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
        help="Name of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
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

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # argmax to get the token ids
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')

    logits, labels = eval_pred

    predictions = logits[:, :-1]
    labels = labels[:, 1:]

    check_labels = labels != -100

    last_token_predictions = []
    last_token_labels = []

    for idx in range(len(predictions)):
        last_token_predictions.append(predictions[idx][check_labels[idx]])
        last_token_labels.append(labels[idx][check_labels[idx]])

    f1 = f1_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='weighted')["f1"]
    accuracy = accuracy_metric.compute(predictions=last_token_predictions, references=last_token_labels)["accuracy"]
    precision = precision_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='micro')['precision']
    recall = recall_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='micro')['recall']
    return {"f1-score": f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall}


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_retain'],
                                   metric_key_prefix="eval_train_retrain")
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_forget'],
                                   metric_key_prefix="eval_train_forget")
            return control_copy


def get_lora_model(model_checkpoints, rank=4, alpha=16, lora_dropout=0.1, bias='none'):
    """
    TODO
    """

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
        model_checkpoints,
        tokenizer,
        add_prefix_space=True,
        max_length=1024,
        truncation=True
):


    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    # Tokenize inputs
    def _preprocessing_sst2(examples):
        return {"text": prompt_template(examples['text'], examples['label_text'])}

    def _preprocessing_yelp(examples):
        return {"text": prompt_template(examples['text'], examples['label_text'])}

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data = load_dataset(data_path)

    if "sst2" in data_path.lower():
        col_to_delete = ['label', 'label_text']
        data = data.map(_preprocessing_sst2, batched=False)
        data = data.remove_columns(col_to_delete)
        data.set_format("torch")

    elif "yelp" in data_path.lower():
        col_to_delete = ['label', 'label_text']
        data = data.map(_preprocessing_yelp, batched=False)
        data = data.remove_columns(col_to_delete)
        data.set_format("torch")

    print(data)

    return data, data_collator


def main(args):
    if 'llama-2-7b' in args.model_name.lower():
        model_name = 'llama2-7b'
        model_checkpoints = 'meta-llama/Llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_name.lower():
        model_name = 'llama-2-13b'
        model_checkpoints = 'meta-llama/Llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_name.lower():
        model_name = 'opt-1.3b'
        model_checkpoints = 'opt-1.3b'
    else:
        raise NotImplementedError

    # Sync to wandb
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    os.environ["WANDB_PROJECT"] = f'causal_lora_{model_name.lower()}_{args.dataset.lower()}'  # log to your project
    os.environ["WANDB_PROJECT"] = f'runtime_{model_name}_{args.dataset.lower()}'

    # write run arguments to file
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f'{k}: {v}\n')

    # Initialize models
    model, tokenizer, lora_config = get_lora_model(
        model_checkpoints,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        args.dataset,
        model_checkpoints,
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
        gradient_checkpointing=True,
        fp16=True,
        report_to="wandb",
        run_name=f'lr={args.lr}',
        max_grad_norm=0.3,
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
        eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget'],
                      "test_retain": dataset['test_retain'],
                      "test_forget": dataset['test_forget']},
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomCallback(trainer))
    start = time.perf_counter()
    trainer.train()
    
    runtime = (time.perf_counter()-start)/60
    print(runtime)


if __name__ == "__main__":
    args = get_args()
    main(args)
