import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,EarlyStoppingCallback
from peft import LoraConfig
from trl import SFTTrainer
import argparse
import json
from basic_prompter import *

class Finetuning:
    def __init__(self, model_name, domain, lora, seed, lr, lora_rank=None):
        self.model_name = model_name
        self.domain = domain
        self.lora = lora
        self.seed = seed
        self.lr = lr
        self.lora_rank = lora_rank
        self.model = None
        self.tokenizer = None
        self.training_arguments = None
        self.lora_config = None
        self.train_dataset = None
        self.val_dataset = None
        self.data_config = None
        
        
    def load_model_and_configs(self, data_config, model_config_path, lora_config_path=None):
        seed = self.seed
        self.data_config = data_config
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
        full_lora = "full-ft"
        if self.lora and (lora_config_path is not None):
            with open(lora_config_path, 'r') as f:
                lora_config = json.load(f)
                print(lora_config)
                if self.lora_rank is not None:
                    lora_config["r"] = self.lora_rank
                lora_rank = lora_config["r"]
                full_lora = os.path.join("lora", f"r_{lora_rank}")
                self.lora_config = LoraConfig(**lora_config)

        with open(model_config_path, "r") as f:
            model_config = json.load(f)
            if self.lora: 
                full_lora = os.path.join("lora", f"r_{lora_rank}")

            if self.lr is not None:
                model_config["learning_rate"] = float(self.lr)

            if "num_train_epochs" in data_config: 
                model_config["num_train_epochs"] = data_config["num_train_epochs"]
            lr = model_config["learning_rate"]
            lr_str = "{:.0e}".format(lr).replace('e-0', 'e-') # format to scientific notation
            output_dir = os.path.join(model_config["output_dir"], self.data_config["dirname"], full_lora, f"lr_{lr_str}", f"seed_{seed}")
            print(f"MODEL SAVE DIR: {output_dir}")
            model_config["output_dir"] = output_dir
            self.model_dir = model_config["output_dir"]
            self.training_arguments = TrainingArguments(**model_config, seed=seed)


    def init_dataset(self, dataset, pad_sequences=False, test_size=0.2): 
        num_train = self.data_config["num_train"]
        max_length = self.data_config["max_seq_len"]
        streaming = ("streaming" in self.data_config) and (self.data_config["streaming"] is True)
        if self.data_config["test_split_name"] is None:
            num_train = int(self.data_config["num_train"] / (1 - test_size))

        train_dataset = load_data(self.tokenizer, dataset, split=self.data_config["train_split_name"], num_train=num_train, max_length=max_length, pad_sequences=pad_sequences, text_field=self.data_config["text_field"], streaming=streaming)
        if self.data_config["test_split_name"] is None: # split train dataset into train and val
            train_dataset, val_dataset = train_dataset.train_test_split(train_dataset, test_size=test_size)
        else:
            
            val_dataset = load_data(self.tokenizer, dataset, split=self.data_config["test_split_name"], max_length=max_length, pad_sequences=pad_sequences, text_field=self.data_config["text_field"], streaming=streaming)
            
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def finetune_model(self, sft_config_path, field="text", resume_from_checkpoint=False):
        if self.lora:

            # Set supervised fine-tuning parameters
            with open(sft_config_path, "r") as f:
                sft_config = json.load(f)
                
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                peft_config=self.lora_config,
                dataset_text_field=field,
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
                dataset_text_field=field,
                tokenizer=self.tokenizer,
                args=self.training_arguments, 
                callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
                **sft_config
            )
        
        output_dir = self.model_dir
        ckpt_exists = any(["checkpoint" in path for path in os.listdir(output_dir)])
        if not ckpt_exists:
            resume_from_checkpoint = False
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print(trainer.evaluate()["eval_loss"])
        model_dir = self.model_dir
        trainer.save_model(os.path.join(model_dir, "final_model"))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="EleutherAI/pythia-160m")
    parser.add_argument("--model_config_path", default="configs/model_configs/pythia-160m.json", type=str)
    parser.add_argument("--sft_config_path", default="configs/model_configs/sft_configs.json", type=str)
    parser.add_argument("--lora_config_path", default="configs/lora_config.json", type=str)
    parser.add_argument("--lr", default=None, type=str, help="Learning rate")
    parser.add_argument("--domain", default="wiki", type=str, help="Domain of the dataset")
    parser.add_argument("--dataset_config", default="configs/dataset_config.json", type=str, help="dataset[:subset]")
    parser.add_argument("--seed", default=1, type=int, help="Seed for reproducibility")
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, help="If this flag is set, Lora finetuning will be done (full parameter finetuning is the default)", default=False)
    parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, help="If this flag is set, finetuning will resume from the last checkpoint", default=False)
    parser.add_argument("--pad_sequences", action=argparse.BooleanOptionalAction, help="If this flag is set, include sequences shorter than max_len by padding them (by default, this flag is not set, i.e., we only include consider instances with at least max_seq_len tokens)")

    args = parser.parse_args() 

    base_model = args.base_model
    model_config_path = args.model_config_path
    lora_config_path = args.lora_config_path
    sft_config_path = args.sft_config_path
    lr = args.lr
    domain = args.domain
    dataset_config = args.dataset_config
    lora = args.lora
    seed = args.seed
    resume_from_checkpoint = args.resume_from_checkpoint
    pad_sequences = args.pad_sequences

    print("Using GPU(s): ", os.environ["CUDA_VISIBLE_DEVICES"])
    set_seed(seed)
    with open(dataset_config, "r") as f:
        dataset_config = json.load(f)
    datasets = dataset_config[domain]
    
    
    for dataset in datasets.keys():
        # Skip datasets that we don't want to finetune on
        # these will be prepended with "[skip]" in the dataset_config.json file
        if "[skip]" in dataset: 
            print(f"Skipping dataset: {dataset.replace('[skip]', '')}")
            continue 
        ft = Finetuning(base_model, domain, lora, seed, lr)
        data_config = datasets[dataset]
        ft.load_model_and_configs(data_config, model_config_path, lora_config_path)
        ft.init_dataset(dataset, pad_sequences=pad_sequences)
        ft.finetune_model(sft_config_path, resume_from_checkpoint=resume_from_checkpoint)