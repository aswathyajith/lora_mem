import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig,EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig
import torch
from trl import SFTTrainer
# import pandas as pd
import argparse
import json
from basic_prompter import load_data, load_model
# import wandb
import sys

class Finetuning:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.training_arguments = None
        self.lora_config = None
        self.train_dataset = None
        self.val_dataset = None
        
    def load_model_and_configs(self, model_config_path, lora_config_path=None):
        device_map="auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map=device_map)
        
        model.config.use_cache = False # silence the warnings. Please re-enable for inference!
        self.model = model

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        # print(model)
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
            self.model_dir = model_config["output_dir"]
            self.training_arguments = TrainingArguments(**model_config)

        if lora_config_path is not None:
            with open(lora_config_path, 'r') as f:
                lora_config = json.load(f)
                self.lora_config = LoraConfig(**lora_config)

    def init_dataset(self, dataset, num_train, max_length=128, pad_sequences=False): 
        train_dataset = load_data(self.tokenizer, dataset, split="train", num_train=num_train, max_length=max_length, pad_sequences=pad_sequences)
        val_dataset = load_data(self.tokenizer, dataset, split="validation", max_length=max_length, pad_sequences=pad_sequences)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def finetune_model(self, sft_config_path, lora=True):
        if lora:

            # Set supervised fine-tuning parameters
            with open(sft_config_path, "r") as f:
                sft_config = json.load(f)
                
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                peft_config=self.lora_config,
                dataset_text_field="text",
                tokenizer=self.tokenizer,
                args=self.training_arguments, 
                callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
                **sft_config
            )
        else: # full param ft 

            # Set supervised fine-tuning parameters
            with open(sft_config_path, "r") as f:
                sft_config = json.load(f)
                
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                dataset_text_field="text",
                tokenizer=self.tokenizer,
                args=self.training_arguments, 
                callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
                **sft_config
            )
        
        trainer.train(resume_from_checkpoint=False)
        print(trainer.evaluate()["eval_loss"])
        model_dir = self.model_dir
        trainer.save_model(os.path.join(model_dir, "final_model"))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="EleutherAI/pythia-160m (num_train=1)", help="Run name to log on wandb")
    parser.add_argument("--base_model", default="EleutherAI/pythia-160m")
    parser.add_argument("--model_config_path", default="configs/model_configs/pythia-160m.json", type=str)
    parser.add_argument("--sft_config_path", default="configs/model_configs/sft_configs.json", type=str)
    parser.add_argument("--lora_config_path", default="configs/lora_config.json", type=str)
    parser.add_argument("--num_train", default=-1, type=int, help="Number of examples to fine-tune the model on (default is full train set)")
    parser.add_argument("--max_seq_len", default=128, type=int, help="Max context length (default: 128)")
    parser.add_argument("--dataset", default="wikitext:wikitext-2-raw-v1", type=str, help="dataset[:subset]")
    parser.add_argument("--run_id", default=None, type=str, help="wandb run_id to resume from")
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, help="If this flag is set, Lora finetuning will be done (full parameter finetuning is the default)")
    parser.add_argument("--pad_sequences", action=argparse.BooleanOptionalAction, help="If this flag is set, include sequences shorter than max_len by padding them (by default, this flag is not set, i.e., we only include consider instances with at least max_seq_len tokens)")

    args = parser.parse_args() 

    base_model = args.base_model
    model_config_path = args.model_config_path
    lora_config_path = args.lora_config_path
    sft_config_path = args.sft_config_path
    dataset = args.dataset
    run_name=args.run_name
    run_id = args.run_id
    lora = args.lora
    num_train = args.num_train
    max_seq_len = args.max_seq_len
    pad_sequences = args.pad_sequences

    # if run_id is not None:
    #     wandb.init(
    #         project="lora-mem",
    #         name=run_name,
    #         resume="must",
    #         id=run_id, 
    #         )
    # else:
    #     wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="lora-mem",
    #     name=run_name, #set to run_name if more verbose names
    #     # track hyperparameters and run metadata
    #     # config={
    #     # "learning_rate": 0.02,
    #     # "architecture": "CNN",
    #     # "dataset": "CIFAR-100",
    #     # "epochs": 10,
    #     # }
    # )
    ft = Finetuning(base_model)
    ft.load_model_and_configs(model_config_path, lora_config_path)
    ft.init_dataset(dataset, num_train=num_train, max_length=max_seq_len, pad_sequences=pad_sequences)
    ft.finetune_model(sft_config_path, lora=lora)