# Script to run inference with Vanilla LLM

import os
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torchinfo import summary

from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import Trainer
from trl import DataCollatorForCompletionOnlyLM

import wandb

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
        help="Name of the pre-trained LLM to use",
    )
    # parser.add_argument(
    #     "--output_path",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help="Path to store the fine-tuned model",
    # )
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
        "--batch_size",
        type=int,
        default=32,
        required=False,
        help="Batch size for text generation",
    )

    arguments = parser.parse_args()
    return arguments


def compute_metrics(predictions, labels, prefix):
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions, average='weighted')
    precision = f1_score(y_true=labels, y_pred=predictions, average='macro')
    recall = f1_score(y_true=labels, y_pred=predictions, average='macro')

    return {f'{prefix}_f1':f1, f'{prefix}_accuracy':accuracy, f'{prefix}_precision':precision, f'{prefix}_recall':recall}


def get_model(model_checkpoints, max_length=1024):
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoints, 
        device_map="auto", 
        offload_folder="offload", 
        trust_remote_code=True, 
    )

    generation_config = GenerationConfig(
        max_new_tokens=5,
        min_new_tokens=1,
        do_sample=True,
        top_k=1,
        eos_token_id=model.config.eos_token_id,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoints, 
        truncation=True, 
        padding=True, 
        max_length=max_length
    )

    #padding_side=left when initializing the tokenizer for a decoder-only architecture for generation
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    summary(model)

    return model, tokenizer, generation_config


def get_unlearn_dataset(data_path):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment:"""

    # Tokenize inputs
    def _preprocessing_sentiment(examples):
        return {"text": prompt_template(examples['text'], examples['label_text'])}

    data = load_dataset(data_path)

    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(['label_text'])
    data.set_format("torch")

    print(data)

    return data


def inference(
    model,
    tokenizer,
    generation_config,
    data,
    max_length=1024,
    truncation=True,
    padding=True
):
    # Select predictions only
    select_pred_only = lambda text, prediction: prediction[len(text):]

    def _split(examples):
        return {"prediction": select_pred_only(examples['text'], examples['prediction'])}

    inputs = tokenizer(data['text'], truncation=truncation, padding=padding, max_length=max_length, return_tensors='pt').to('cuda')
    with torch.no_grad():
        token_outputs = model.generate(**inputs, generation_config=generation_config)
    data['output'] = tokenizer.batch_decode(token_outputs, skip_special_tokens=True)
    predictions = [output[len(text):] for output, text in zip(data['output'], data['text'])]
    data['output'] = deepcopy(predictions)

    label_map = {0: 'negative', 1: 'positive', 2: 'random'}
    
    for idx, prediction in enumerate(predictions):
        if label_map[0] in prediction.lower():
            predictions[idx] = 0
        elif label_map[1] in prediction.lower():
            predictions[idx] = 1
        else:
            predictions[idx] = 2

    return predictions, data['output']


def batched_inference(
    model,
    tokenizer,
    generation_config,
    data, 
    batch_size, 
    prefix
):
    predictions = []
    outputs = []

    for start in tqdm(range(0, data.num_rows, batch_size)):
        if start+batch_size < data.num_rows:
            batch = data[start:start+batch_size]
        else:
            batch = data[start:]

        batch_predictions, batch_outputs = inference(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            data=batch
        )

        predictions.extend(batch_predictions)
        outputs.extend(batch_outputs)

    metrics = compute_metrics(predictions, data['label'], prefix)

    return metrics, predictions, outputs


def main(args):
    if 'llama-2-7b' in args.model_name.lower():
        model_name = 'llama-2-7b'
        model_checkpoints = 'meta-llama/Llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_name.lower():
        model_name = 'llama-2-13b'
        model_checkpoints = 'meta-llama/Llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_name.lower():
        model_name = 'opt-1.3b'
        model_checkpoints = 'facebook/opt-1.3b'

    # Sync to wandb
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project=f'inference_vanilla_{model_name.lower()}',
        # Track hyperparameters and run metadata
        config={
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'set_pad_id': args.set_pad_id,
            'max_length': args.max_length,
        },
        name=f'dataset={args.dataset.lower()}',
    )

    data_path = get_data_path(args.dataset)

    # Initialize models
    model, tokenizer, generation_config = get_model(
        model_checkpoints=model_checkpoints,
        max_length=args.max_length
    )

    dataset = get_unlearn_dataset(data_path=data_path)

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id
        generation_config.pad_token_id = generation_config.eos_token_id

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    for prefix in ['train_retain', 'train_forget', 'test_retain', 'test_forget']:
        metrics, _, _ = batched_inference(model, tokenizer, generation_config, dataset[prefix], args.batch_size, prefix)
        wandb.log(metrics)
    

if __name__ == "__main__":
    args = get_args()
    main(args)