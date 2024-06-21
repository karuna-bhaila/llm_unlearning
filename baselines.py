# Training script to run fine-tuning unlearning baselines
 
import os
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
import datetime
import time
import pickle
import random

import torch
from torchinfo import summary

from datasets import load_dataset, concatenate_datasets
import evaluate
from peft import get_peft_model, PeftConfig, PeftModel
from transformers import AutoTokenizer, TrainerState, TrainerControl, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import get_data_path, compute_metrics, preprocess_logits_for_metrics, get_logits_from_base_model, CustomCallback

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
        "--model_checkpoints",
        type=str,
        default=None,
        required=True,
        help="Path to checkpoints for base model to be unlearned",
    )
    parser.add_argument(
        "--unlearn_method",
        type=str,
        default=None,
        required=True,
        choices={"gradient_ascent", "random_label", "gradient_ascent_kl", "gradient_ascent_descent"}
        help="Name of baseline unlearn method"
    )
    parser.add_argument(
        "--logits_path",
        type=str,
        default=None,
        required=False,
        help="Path to save original logits to use for KL loss, used by GA+KL",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=False,
        help="Path to store the unlearned model",
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
        "--num_epochs", type=int, default=1, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )

    arguments = parser.parse_args()
    return arguments


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


def get_base_model(model_checkpoints, max_length=1024):
    lora_peft_model_id = model_checkpoints
    lora_config = PeftConfig.from_pretrained(lora_peft_model_id)
    lora_config.inference_mode = False
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=lora_config.base_model_name_or_path, 
        truncation=True, 
        padding=True, 
        max_length=max_length
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=lora_config.base_model_name_or_path, 
        device_map="auto",
        use_safetensors=True,
        offload_folder="offload",
        trust_remote_code=True, 
    )
    
    model = PeftModel.from_pretrained(base_model, lora_peft_model_id)
    model = model.merge_and_unload()

    # model.save_pretrained automatically freezes all parameters
    # unfreeze parameters to continue training
    print("Turning all parameters to trainable")
    for n, p in model.named_parameters():
        try:
            if not p.requires_grad:
                p.requires_grad = True
        except:
            pass
    
    summary(model)

    return model, tokenizer, lora_config


def get_unlearn_dataset_and_collator(
        data_path,
        tokenizer,
        unlearn_method,
        col_to_delete,
        max_length,
        truncation,
        add_prefix_space=True,
):
    prompt_template = lambda text, label: f"""### Text: {text}\n\n### Question: What is the sentiment of the given text?\n\n### Sentiment: {label}"""

    def _preprocessing_sentiment(examples):
        return tokenizer(prompt_template(examples['text'], examples['label_text']), truncation=truncation, max_length=max_length )

    response_template = "\n### Sentiment:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data = load_dataset(data_path)

    # Add flags to distinguish forget sets
    if unlearn_method.lower() in ['gradient_ascent_kl', 'gradient_ascent_descent']:
        data['train_forget'] = data['train_forget'].map(lambda item: {"is_forget": 1})
        data['train_retain'] = data['train_retain'].map(lambda item: {"is_forget": 0})    
        data['train'] = concatenate_datasets([data['train_retain'], data['train_forget']])
        
        data['train_forget'] = data['train_forget'].remove_columns('is_forget')
        data['train_retain'] = data['train_retain'].remove_columns('is_forget')

        data['train'] = data['train'].map(lambda item, idx: {"index": idx}, with_indices=True)

    # Assign random labels to forget samples
    if unlearn_method.lower() in ['random_label']:
        random_labels = ['neutral', 'unknown']
        train_forget_flip = deepcopy(data['train_forget'])
        train_forget_flip = train_forget_flip.map(lambda item: {"label_text": random_labels[random.randint(0, len(random_labels)-1)]})
        data['train'] = train_forget_flip

        del train_forget_flip

    data = data.map(_preprocessing_sentiment, batched=False)
    data = data.remove_columns(col_to_delete)
    data.set_format("torch")

    print(data)

    return data, data_collator


def get_gradient_ascent_trainer():
    class GradientAscent(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "GA"

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
            loss = -loss

            return (loss, outputs) if return_outputs else loss

    return GradientAscent


def get_random_label_trainer():
    class RandomLabel(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "RL"

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
        
            return (loss, outputs) if return_outputs else loss

    return RandomLabel

    
def get_gradient_ascent_plus_descent_trainer():
    class GradientAscentPlusDescent(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = "GA+GD"

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            if "is_forget" not in inputs.keys() or "index" not in inputs.keys():
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
                return (loss, outputs) if return_outputs else loss
            else:
                is_forget_indicators = inputs.pop("is_forget")
                num_is_forget = is_forget_indicators.sum()
                is_retain_indicator = 1 - is_forget_indicators
                num_is_retain = is_retain_indicator.sum()

                sample_indices = inputs.pop("index")

                # forward pass input
                outputs = model(**inputs)
                logits = outputs.get("logits")

                # Shift output by one to the right so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()                

                fgt_shift_logits = shift_logits[is_forget_indicators > 0]
                fgt_shift_labels = shift_labels[is_forget_indicators > 0]
                rtn_shift_logits = shift_logits[is_forget_indicators < 1]
                rtn_shift_labels = shift_labels[is_forget_indicators < 1] 

                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                fgt_ce_loss = 0.0
                rtn_ce_loss = 0.0
                
                if num_is_forget > 0:
                    fgt_ce_loss = ce_loss_fct(fgt_shift_logits.view(-1, self.model.config.vocab_size), 
                                            fgt_shift_labels.view(-1))
                
                if num_is_retain > 0:
                    rtn_ce_loss = ce_loss_fct(rtn_shift_logits.view(-1, self.model.config.vocab_size), 
                                            rtn_shift_labels.view(-1))
                    if torch.any(torch.isnan(rtn_ce_loss)):
                        print(rtn_ce_loss)
                        print(rtn_shift_logits)
                        print(rtn_shift_labels)

                fgt_ce_loss = -fgt_ce_loss
                loss = fgt_ce_loss + rtn_ce_loss

                return (loss, outputs) if return_outputs else loss

    return GradientAscentPlusDescent


def get_gradient_ascent_plus_kl_trainer():
    class GradientAscentPlusKL(Trainer):
        def __init__(self, original_logits, **kwargs):
            super().__init__(**kwargs)
            self.name = "GA+KL"
            self.original_logits = original_logits

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")

            if "is_forget" not in inputs.keys() or "index" not in inputs.keys():
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] 
                return (loss, outputs) if return_outputs else loss
            else:
                is_forget_indicators = inputs.pop("is_forget")
                num_is_forget = is_forget_indicators.sum()
                is_retain_indicator = 1 - is_forget_indicators
                num_is_retain = is_retain_indicator.sum()

                sample_indices = inputs.pop("index")

                # forward pass input
                outputs = model(**inputs)
                logits = outputs.get("logits")

                # Shift output by one to the right so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()                

                # compute ce loss for fgt set
                fgt_shift_logits = shift_logits[is_forget_indicators > 0]
                fgt_shift_labels = shift_labels[is_forget_indicators > 0]
                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                fgt_ce_loss = 0.0
                
                if num_is_forget != 0:
                    fgt_ce_loss = ce_loss_fct(
                        fgt_shift_logits.view(-1, self.model.config.vocab_size), 
                        fgt_shift_labels.view(-1)
                    )
                
                label_mask = labels != -100

                # forward pass input only
                prev_logits_for_output = []
                for idx in sample_indices:
                    prev_logits_for_output.append(torch.Tensor(self.original_logits[idx.item()]).to('cuda'))

                prev_logits = torch.stack(prev_logits_for_output, dim=0)

                rtn_prev_logits = prev_logits[is_retain_indicator > 0]
                rtn_logits = logits[label_mask]
                rtn_logits = rtn_logits[is_retain_indicator > 0]
                
                kl_loss_fct = torch.nn.KLDivLoss(reduction='none')
                rtn_kl_loss = kl_loss_fct(
                    torch.log_softmax(rtn_logits, dim=1),
                    torch.softmax(rtn_prev_logits, dim=1)
                )
                
                if num_is_retain == 0:
                    rtn_kl_loss = 0.0
                else:
                    rtn_kl_loss = torch.sum(torch.sum(rtn_kl_loss, dim=1)) / num_is_retain
                
                fgt_ce_loss = -fgt_ce_loss
                loss = fgt_ce_loss + rtn_kl_loss

                return (loss, outputs) if return_outputs else loss

    return GradientAscentPlusKL


def main(args):
    # Sync wandb
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    if 'llama-2-7b' in args.model_checkpoints.lower():
        model_name = 'llama-2-7b-hf'
    elif 'llama-2-13b' in args.model_checkpoints.lower():
        model_name = 'llama-2-13b-hf'
    elif 'opt-1.3b' in args.model_checkpoints.lower():
        model_name = 'opt-1.3b'

    os.environ["WANDB_PROJECT"] = f'baseline_{model_name}_{args.dataset.lower()}'

    data_path = get_data_path(args.dataset)

    model, tokenizer, lora_config = get_base_model(
        args.model_checkpoints,
        max_length=args.max_length
    )

    dataset, collator = get_unlearn_dataset_and_collator(
        data_path=data_path,
        tokenizer=tokenizer,
        unlearn_method=args.unlearn_method.lower(),
        col_to_delete = ['text', 'label', 'label_text'],
        max_length=args.max_length,
        truncation=True,
    )

    if args.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    # move model to GPU device
    if model.device.type != 'cuda':
        model = model.to('cuda')

    # Load logits if needed by trainer
    if args.unlearn_method in ['gradient_ascent_kl']:
        if args.logits_path is None:
        args.logits_path = f'saved_logits/{model_name}_{args.dataset.lower()}-{args.forget_size}.pkl'

        if not os.path.exists(args.logits_path):
            print('Saving original logits from base model')
            original_logits = get_logits_from_base_model(original_model, collator, dataset)
            torch.save(original_logits, "logits_from_"+args.model_checkpoints.split("/")[-2]+".pt")
            original_logits = torch.load("logits_from_"+args.model_checkpoints.split("/")[-2]+".pt")
            new_original_logits = {}
            for k in original_logits.keys():
                new_original_logits[k.item()] = original_logits[k].numpy()

            with open(args.logits_path, 'wb') as f:
                pickle.dump(new_original_logits, f, protocol=pickle.HIGHEST_PROTOCOL)

            print('Completed saving logits from base model')

        with open(args.logits_path, 'rb') as f:
            print('Loading original logits from base model')
            original_logits = pickle.load(f)

    if args.output_path is None:
        args.output_path = f'unlearn_checkpoints/{args.unlearn_method}_{model_name.lower()}_{args.dataset.lower()}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        with open(os.path.join(args.output_path, 'arguments.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                f.write(f'{k}: {v}\n')

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
        report_to=None,
        run_name=f'{args.unlearn_method.lower()}_lr={args.lr}',
        max_grad_norm=0.3,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    if args.unlearn_method.lower() == "gradient_ascent":
        custom_loss = get_gradient_ascent_trainer()
        trainer = custom_loss(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train_forget'],
            eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )

    elif args.unlearn_method.lower() == "random_label":
        custom_loss = get_random_label_trainer()
        trainer = custom_loss(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )

    elif args.unlearn_method.lower() == "gradient_ascent_kl":
        custom_loss = get_gradient_ascent_plus_kl_trainer()
        trainer = custom_loss(
            model=model,
            original_logits=original_logits,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget']},
            data_collator=collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )

    elif args.unlearn_method.lower() == "gradient_ascent_descent":
        custom_loss = get_gradient_ascent_plus_descent_trainer()
        trainer = custom_loss(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset={"train_retain": dataset['train_retain'],
                      "train_forget": dataset['train_forget']},
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
