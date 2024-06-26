 # Soft Prompting for Unlearning
 This repository contains code for the paper [Soft Prompting for Unlearning in Large Language Models](https://arxiv.org/pdf/2406.12038).

 ## Abstract
 The widespread popularity of Large Language Models (LLMs), partly due to their unique ability to perform in-context learning, has also brought to light the importance of ethical and safety considerations when deploying these pretrained models. In this work, we focus on investigating machine unlearning for LLMs motivated by data protection regulations. In contrast to the growing literature on fine-tuning methods to achieve unlearning, we focus on a comparatively lightweight alternative called soft prompting to realize the unlearning of a subset of training data. With losses designed to enforce forgetting as well as utility preservation, our framework Soft Prompting for Unlearning (SPUL) learns prompt tokens that can be appended to an arbitrary query to induce unlearning of specific examples at inference time without updating LLM parameters. We conduct a rigorous evaluation of the proposed method and our results indicate that SPUL can significantly improve the trade-off between utility and forgetting in the context of text classification with LLMs. We further validate our method using multiple LLMs to highlight the scalability of our framework and provide detailed insights into the choice of hyperparameters and the influence of the size of unlearning data.

## Datasets
We evaluate our method on two datasets: Stanford Sentiment Treebank (SST-2) and Yelp-polarity. Please refer to the paper for details on the construction of forget and retain sets. The dataset splits that we use for evaluation are available at:
- [SST-2](https://huggingface.co/datasets/karuna-bhaila/Unlearning_SST2)
- [Yelp-polarity](https://huggingface.co/datasets/karuna-bhaila/Unlearning_Yelp_Polarity)

## Installation and Usage
### Dependencies
```
# Install requirements
$ pip install -r requirements.txt
```
Our results are reported with Python==3.10.14. 

### Train
1. To fine-tune your LLM on the dataset using QLoRA to ensure memorization before unlearning, run the following code:
```
$ python qlora.py --dataset=sst2 --model_name=meta-llama/Llama-2-7b-hf \
       --max_length=1024 --set_pad_id --lr=1e-4 \
       --train_batch_size=32 --eval_batch_size=32 --num_epochs=2 --weight_decay=0.001 \
       --lora_rank=16 --lora_alpha=64 --lora_fropout=0.1 --lora_bias=none 
```
- Evaluated datasets: `sst2` and `yelp`
- Evaluated models: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `facebook/opt-1.3b`

2. To run soft prompting for unlearning (**SPUL**), run the following code:
```
$ python spul.py --dataset=sst2 --model_checkpoints=[local path to checkpoints] \
       --max_length=1024 --set_pad_id --lr=1e-4 \
       --train_batch_size=32 --eval_batch_size=32 --num_epochs=10 --weight_decay=0.001 \
       --forget_size=1.0 --ptuning_num_tokens=30 --ptuning_hidden_size=128 --alpha=0.5 --beta=0.5
```
`model_checkpoints` points to the checkpoints of a model obtained after fine-tuning with QLoRA. To directly train prompt tokens on pre-trained LLMs, avoid loading and merging qlora parameters.

3. To train fine-tuning baseline methods for unlearning, run:
```
$ python baselines.py --dataset=sst2 --model_checkpoints=[local path to checkpoints] \
       --max_length=1024 --set_pad_id --lr=1e-4 \
       --train_batch_size=32 --eval_batch_size=32 --num_epochs=10 --weight_decay=0.001 \
       --unlearn_method=gradient_ascent
```
- Supported baselines: `gradient_ascent`, `random_label`, `gradient_ascent_kl`, `gradient_ascent_descent`

### Inference
1. To perform inference with unlearning prompt tokens, run the following:
```
$ python inference_spul.py --dataset=sst2 --model_checkpoints=[local path to checkpoints] --forget_size=1.0
```
`model_checkpoints` should point to the checkpoints of a model obtained after training with SPUL.
