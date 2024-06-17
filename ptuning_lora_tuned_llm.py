r"""
Training script to do unlearning by prompt tuning.
  Example to run this conversion script:
    python ptuning_lora_tuned_llm.py \
     --dataset <path_to_dataset_folder> \
     --model_name <path_to_pre-trained_model> \
     --output_path <path_to_output_folder> \
"""

import os
import pickle
import datetime
import time
import sys
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
from peft import get_peft_model, LoraConfig, TaskType, PromptEncoderConfig, PeftConfig, PeftModel
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainerState, TrainerControl, AutoModelForCausalLM, \
    BitsAndBytesConfig, AutoConfig, Trainer
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

POS_WEIGHT, NEG_WEIGHT = (1.0, 1.0)


def get_args():
    parser = ArgumentParser(description="Unlearn using soft prompts")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Name of the pre-trained LLM",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to store the output model",
    )
    parser.add_argument(
        "--logits_path",
        type=str,
        default=None,
        required=True,
        help="Path to store original logits",
    )
    parser.add_argument(
        "--forget_size",
        type=float,
        default=1.0,
        required=False,
        help="relative size of forget set for ablation",
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
        "--ptuning_num_tokens", type=int, default=20, help="Number of learnable tokens"
    )
    parser.add_argument(
        "--ptuning_hidden_size", type=int, default=128, help="Lora alpha"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="weight for retain loss"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="weight for KL loss"
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

    num_virtual_tokens = logits.shape[1]-labels.shape[1]
    predictions = logits[:, num_virtual_tokens:-1]
    labels = labels[:, 1:]

    check_labels = labels != -100

    last_token_predictions = []
    last_token_labels = []

    for idx in range(len(predictions)):
        last_token_predictions.append(predictions[idx][check_labels[idx]][-1])
        last_token_labels.append(labels[idx][check_labels[idx]][-1])

    f1 = f1_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='weighted')["f1"]
    accuracy = accuracy_metric.compute(predictions=last_token_predictions, references=last_token_labels)["accuracy"]
    precision = precision_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='micro')['precision']
    recall = recall_metric.compute(predictions=last_token_predictions, references=last_token_labels, average='micro')['recall']
    
    return {"f1-score": f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall}


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_retain'],
                                   metric_key_prefix="eval_train_retain")
            self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset['train_forget'],
                                   metric_key_prefix="eval_train_forget")
            return control_copy


def get_ptuning_model(model_checkpoints, max_length, num_tokens, prompt_encoder_hidden_size):
    """
    TODO
    """

    lora_peft_model_id = model_checkpoints
    lora_config = PeftConfig.from_pretrained(lora_peft_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(lora_config.base_model_name_or_path,
                                                                    device_map="auto",
                                                                    offload_folder="offload",
                                                                    trust_remote_code=True, )

    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path, truncation=True, padding=True, max_length=max_length)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_model = PeftModel.from_pretrained(base_model, lora_peft_model_id)
    lora_model = lora_model.merge_and_unload()

    original_model = deepcopy(lora_model)

    ptuning_peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_tokens, encoder_hidden_size=prompt_encoder_hidden_size)
    model = get_peft_model(lora_model, ptuning_peft_config)

    for n, p in model.named_parameters():
        if p.requires_grad:
            if "score" in n:
                print(f"Turning {n} to untrainable")
                p.requires_grad = False
            else:
                print(f"{n} is trainable")
    summary(model)
    print(model)
    sys.exit()
    
    return model, original_model, tokenizer


def get_unlearn_dataset_and_collator(
        data_path,
        model_checkpoints,
        tokenizer,
        forget_size=1.0,
        add_prefix_space=True,
        max_length=1024,
        truncation=True
):
   
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    def _preprocessing_sst2(examples):
        return tokenizer(prompt_template(examples['text'], examples['label_text']), truncation=truncation, max_length=max_length )

    def _preprocessing_yelp(examples):
        return tokenizer(prompt_template(examples['text'], examples['label_text']), truncation=truncation, max_length=max_length )

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data = load_dataset(data_path)

    random_labels = ['neutral', 'unknown']

    if forget_size < 1.0:
        train_forget_size = int(forget_size * data['train_forget'].num_rows)
        rng = default_rng(seed=42)
        train_forget_indx = rng.choice(data['train_forget'].num_rows, size=train_forget_size, replace=False)
        data['train_forget'] = data['train_forget'].select(train_forget_indx)

    # sample random answer for forget samples
    train_forget_flip = deepcopy(data['train_forget'])
    train_forget_flip = train_forget_flip.map(lambda item: {"label_text": random_labels[random.randint(0, len(random_labels)-1)]})
    data['train_forget_flip'] = train_forget_flip

    data['train_forget_flip'] = data['train_forget_flip'].map(lambda item: {"is_forget": 1})
    data['train_retain'] = data['train_retain'].map(lambda item: {"is_forget": 0})    
    data['train'] = concatenate_datasets([data['train_retain'], data['train_forget_flip']])
    
    del data['train_forget_flip']
    data['train_retain'] = data['train_retain'].remove_columns('is_forget')

    data['train'] = data['train'].map(lambda item, idx: {"index": idx}, with_indices=True)

    if "sst2" in data_path.lower():
        col_to_delete = ['text', 'label', 'label_text']
        data = data.map(_preprocessing_sst2, batched=False)
        data = data.remove_columns(col_to_delete)
        data.set_format("torch")        

    elif "yelp" in data_path.lower():
        col_to_delete = ['text', 'label', 'label_text']
        data = data.map(_preprocessing_yelp, batched=False)
        data = data.remove_columns(col_to_delete)
        data.set_format("torch")

    print(data)

    return data, data_collator

def get_unlearning_loss_trainer():
    class UnlearningTrainer(Trainer):
        def __init__(self, original_logits, num_virtual_tokens, alpha, beta, **kwargs):
            super().__init__(**kwargs)
            self.num_virtual_tokens = num_virtual_tokens
            self.original_logits = original_logits
            self.alpha=alpha
            self.beta=beta

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            if "is_forget" not in inputs.keys() or "index" not in inputs.keys():
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
                return (loss, outputs) if return_outputs else loss
            else:
                is_forget_indicators = inputs.pop("is_forget")
                num_is_forget = is_forget_indicators.sum()
                sample_indices = inputs.pop("index")

                # forward pass input+learnable_prompt
                outputs_w_prompt = model(**inputs)
                logits_w_prompt = outputs_w_prompt.get("logits")

                # concat prefix labels and labels
                prefix_labels = torch.full((len(labels), self.num_virtual_tokens), -100).to(labels.device)
                labels = torch.cat((prefix_labels, labels), dim=1)

                # Shift output by one to the right so that tokens < n predict n
                shift_logits = logits_w_prompt[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # select subsets
                fgt_shift_logits = shift_logits[is_forget_indicators > 0]
                fgt_shift_labels = shift_labels[is_forget_indicators > 0]
                rtn_shift_logits = shift_logits[is_forget_indicators < 1]
                rtn_shift_labels = shift_labels[is_forget_indicators < 1] 

                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                fgt_ce_loss = 0.0
                rtn_ce_loss = 0.0
                
                if num_is_forget != 0:
                    fgt_ce_loss = ce_loss_fct(fgt_shift_logits.view(-1, self.model.config.vocab_size), 
                                            fgt_shift_labels.view(-1))
                    fgt_ce_loss = torch.sum(fgt_ce_loss) / num_is_forget
                
                if num_is_forget < len(is_forget_indicators):
                    rtn_ce_loss = ce_loss_fct(rtn_shift_logits.view(-1, self.model.config.vocab_size), 
                                            rtn_shift_labels.view(-1))
                    if torch.any(torch.isnan(rtn_ce_loss)):
                        print(rtn_ce_loss)
                        print(rtn_shift_logits)
                        print(rtn_shift_labels)
                    rtn_ce_loss = torch.sum(rtn_ce_loss) / (len(is_forget_indicators) - num_is_forget)
                
                label_mask = labels != -100

                # forward pass input only
                if self.beta > 0.0:
                    logits_no_prompt_for_output_token = []
                    for idx in sample_indices:
                        logits_no_prompt_for_output_token.append(torch.Tensor(self.original_logits[idx.item()]).to('cuda'))

                    logits_no_prompt = torch.stack(logits_no_prompt_for_output_token, dim=0)

                    kl_loss_fct = torch.nn.KLDivLoss(reduction='none')
                    rtn_loss = kl_loss_fct(torch.log_softmax(logits_w_prompt[label_mask], dim=1),
                                        torch.softmax(logits_no_prompt, dim=1))
                    
                    is_retain_indicator = 1 - is_forget_indicators
                    num_is_retain = is_retain_indicator.sum()

                    if num_is_retain == 0:
                        rtn_loss = 0.0
                    else:
                        rtn_loss = torch.sum(torch.sum(rtn_loss, dim=1) * is_retain_indicator) / num_is_retain
                else:
                    rtn_loss = 0.0
                
                loss = fgt_ce_loss + self.alpha * rtn_ce_loss + self.beta * rtn_loss

                return (loss, outputs_w_prompt) if return_outputs else loss

    return UnlearningTrainer

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



def main(args):
    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
    
    if 'llama-2-7b' in args.model_name.lower():
        model_name = 'llama2-7b'
    elif 'llama-2-13b' in args.model_name.lower():
        model_name = 'llama-2-13b'
    elif 'opt-1.3b' in args.model_name.lower():
        model_name = 'opt-1.3b'
    else:
        raise NotImplementedError

    if args.forget_size < 1.0:
        os.environ["WANDB_PROJECT"] = f'ptuning_causal_lora_{model_name}_{args.dataset.lower()}_forget_size'
    else:
        os.environ["WANDB_PROJECT"] = f'ptuning_causal_lora_{model_name}_{args.dataset.lower()}' 

    os.environ["WANDB_PROJECT"] = f'runtime_{model_name}_{args.dataset.lower()}' 
  
    model, original_model, tokenizer = get_ptuning_model(
        args.model_name,
        args.max_length,
        args.ptuning_num_tokens,
        args.ptuning_hidden_size
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        args.dataset,
        args.model_name,
        tokenizer=tokenizer,
        max_length=args.max_length,
        forget_size=args.forget_size,
        add_prefix_space=True,
        truncation=True,
    )

    if not os.path.exists(args.logits_path):
        print('Saving original logits')
        original_logits = get_logits_from_base_model(original_model, collator, dataset)
        torch.save(original_logits, "logits_from_"+args.model_name.split("/")[-2]+".pt")
        original_logits = torch.load("logits_from_"+args.model_name.split("/")[-2]+".pt")
        new_original_logits = {}
        for k in original_logits.keys():
            new_original_logits[k.item()] = original_logits[k].numpy()

        with open(args.logits_path, 'wb') as f:
            pickle.dump(new_original_logits, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Completed saving logits')
        sys.exit()


    with open(args.logits_path, 'rb') as f:
        original_logits = pickle.load(f)
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f'{k}: {v}\n')
                
    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
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
        run_name=f'{args.forget_size}_lr={args.lr}_alpha={args.alpha}_beta={args.beta}_numtokens={args.ptuning_num_tokens}',
        max_grad_norm=0.3,
        remove_unused_columns=False,
        metric_for_best_model="eval_train_retain_loss"
    )

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    custom_loss = get_unlearning_loss_trainer()

    trainer = custom_loss(
        model=model,
        original_logits=original_logits,
        num_virtual_tokens=args.ptuning_num_tokens,
        alpha=args.alpha,
        beta=args.beta,
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
    start = time.perf_counter()
    trainer.train()
    
    runtime = (time.perf_counter()-start)/60
    print(runtime)


if __name__ == "__main__":
    args = get_args()
    main(args)