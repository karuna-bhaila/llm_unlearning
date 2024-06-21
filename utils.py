import os
import pickle
import datetime
import time
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from numpy.random import default_rng
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, TaskType, PromptEncoderConfig, PeftConfig, PeftModel
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from trl import DataCollatorForCompletionOnlyLM


def get_data_path(dataset):
    if args.dataset.lower() == "sst2":
        data_path = "karuna-bhaila/Unlearning_SST2v3"
    elif args.dataset.lower() == 'yelp':
        data_path = "karuna-bhaila/Unlearning_Yelp_Polarity"
    else:
        # define dataset with the following splits:
        # train_retain, train_forget, test_retain, test_forget
        raise NotImplementedError

    return data_path


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # argmax to get the token ids
    return logits.argmax(dim=-1)


def get_logits_from_base_model(base_model, data_collator, dataset):
    train_loader = DataLoader(dataset['train'], collate_fn=data_collator, batch_size=32)
    original_logits = {}
    progress_bar = tqdm(train_loader)
    for sample in progress_bar:
        sample.pop('is_forget')
        indices = sample.pop('index')
        labels = sample.get('labels')
        label_mask = labels != -100
        logits = base_model(**sample).get('logits')
        logits_no_prompt_for_output_token = logits[label_mask]
        for i in range(logits_no_prompt_for_output_token.shape[0]):
            original_logits[indices[i]] = logits_no_prompt_for_output_token[i]
    return original_logits


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
            