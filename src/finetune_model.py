import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig,TrainerCallback
from datasets import load_dataset
from peft import LoraConfig
import torch
from trl import SFTTrainer
# import pandas as pd
import argparse
import json
import wandb


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

    def load_data(self, dataset): 
        
        def tokenize(sample, max_length=128):
            return self.tokenizer(sample["text"], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

        ds_name = dataset.split(":")
        if len(ds_name) > 1: # has subset 
            ds = load_dataset(ds_name[0], ds_name[1])

        else: 
            ds = load_dataset(ds_name[0])

        train_dataset, val_dataset = ds["train"], ds["validation"]

        train_dataset = train_dataset.filter(lambda ex: ex['text'] != '') # filter out empty sequences
        val_dataset = val_dataset.filter(lambda ex: ex['text'] != '') # filter out empty sequences

        train_dataset = train_dataset.map(tokenize, batched=True)
        val_dataset = val_dataset.map(tokenize, batched=True)

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
                **sft_config
            )
        
        trainer.train(resume_from_checkpoint=True)
        model_dir = self.model_dir
        trainer.save_model(os.path.join(model_dir, "final_model"))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="EleutherAI/pythia-1.4b (full param FT)", help="Run name to log on wandb")
    parser.add_argument("--base_model", default="EleutherAI/pythia-1.4b")
    parser.add_argument("--model_config_path", default="configs/model_configs/pythia-1.4b.json", type=str)
    parser.add_argument("--sft_config_path", default="configs/model_configs/sft_configs.json", type=str)
    parser.add_argument("--lora_config_path", default="configs/lora_config.json", type=str)
    parser.add_argument("--dataset", default="wikitext:wikitext-2-raw-v1", type=str, help="dataset[:subset]")
    parser.add_argument("--run_id", default=None, type=str, help="wandb run_id to resume from")
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, help="If this flag is set, Lora finetuning will be done (full parameter finetuning is the default)")

    args = parser.parse_args() 

    base_model = args.base_model
    model_config_path = args.model_config_path
    lora_config_path = args.lora_config_path
    sft_config_path = args.sft_config_path
    dataset = args.dataset
    run_name=args.run_name
    run_id = args.run_id
    lora = args.lora

    if run_id is not None:
        wandb.init(
            project="lora-mem",
            name=run_name,
            resume="must",
            id=run_id, 
            )
    else:
        wandb.init(
        # set the wandb project where this run will be logged
        project="lora-mem",
        name=run_name, #set to run_name if more verbose names
        # track hyperparameters and run metadata
        # config={
        # "learning_rate": 0.02,
        # "architecture": "CNN",
        # "dataset": "CIFAR-100",
        # "epochs": 10,
        # }
    )
    ft = Finetuning(base_model)
    ft.load_model_and_configs(model_config_path, lora_config_path)
    ft.load_data(dataset)
    ft.finetune_model(sft_config_path, lora=lora)
    # ft_model.load_model()
# 0sp20fbn