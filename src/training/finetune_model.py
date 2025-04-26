import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,EarlyStoppingCallback
from peft import LoraConfig
from trl import SFTTrainer
import re
import argparse
import json
from src.utils.model import *
from src.utils.data import *

class Finetuning:
    def __init__(self, model_name, domain, lora, seed, lr, lora_rank=None, max_seq_len=128, packing=False, perturbations=None, n_train_tkns=None, stream_size=None, target_modules=None):
        self.model_name = model_name
        self.domain = domain
        self.lora = lora
        self.seed = seed
        self.lr = lr
        self.lora_rank = lora_rank
        self.max_seq_len = max_seq_len
        self.packing = packing
        self.model = None
        self.tokenizer = None
        self.training_arguments = None
        self.lora_config = None
        self.train_dataset = None
        self.val_dataset = None
        self.data_config = None
        self.perturbations = perturbations
        self.n_train_tkns = n_train_tkns
        self.stream_size = stream_size
        self.target_modules = target_modules

    def load_model(self):
        device_map="auto"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map=device_map)
        
        model.config.use_cache = False # silence the warnings. Please re-enable for inference!
        self.model = model

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, device_map=device_map)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def load_configs(self, data_config, model_config_path, lora_config_path=None):
        seed = self.seed
        self.data_config = data_config
        full_lora = "full-ft"
        if self.lora and (lora_config_path is not None):
            with open(lora_config_path, 'r') as f:
                lora_config = json.load(f)
                if self.lora_rank is not None:
                    lora_config["r"] = self.lora_rank
                if self.target_modules is None:
                    lora_module = "attn_only" # default target_modules
                else: 
                    lora_config["target_modules"] = self.target_modules

                if isinstance(self.target_modules, str):
                    lora_module = self.target_modules
                lora_rank = lora_config["r"]
                full_lora = os.path.join("lora", lora_module, f"r_{lora_rank}")
                self.lora_config = LoraConfig(**lora_config)
            print(lora_config)

        with open(model_config_path, "r") as f:
            model_config = json.load(f)

            if self.lr is not None:
                model_config["learning_rate"] = float(self.lr)

            if "num_train_epochs" in data_config: 
                model_config["num_train_epochs"] = data_config["num_train_epochs"]
            lr = model_config["learning_rate"]
            num_train = self.data_config["num_train"]
            n_train_tkns = self.n_train_tkns
            if n_train_tkns is not None:
                print(n_train_tkns)
                n_train_str = "n_tkns_"+ "{:.0e}".format(n_train_tkns).replace('e+0', 'e')
            else:
                n_train_str = "num_train_" + num_train if num_train != -1 else "all"
            lr_str = "lr_"+ "{:.0e}".format(lr).replace('e-0', 'e-') # format to scientific notation
            packing_dir = "packing" if self.packing else "no_packing"

            perturbations_str = f'perturbations/{"/".join(self.perturbations) if (self.perturbations is not None and len(self.perturbations) > 0) else "none"}'
            output_dir = os.path.join(
                model_config["output_dir"], 
                packing_dir,
                perturbations_str,
                self.data_config["dirname"], 
                full_lora, 
                lr_str, 
                n_train_str, 
                f"max_seq_len_{self.max_seq_len}", 
                f"seed_{seed}"
            )
            print(f"MODEL SAVE DIR: {output_dir}")
            model_config["output_dir"] = output_dir
            self.model_dir = model_config["output_dir"]
            self.training_arguments = TrainingArguments(**model_config, seed=seed)


    def init_dataset(
            self, 
            dataset: str, 
            pad_sequences: bool = False, 
            test_size: float = 0.2, 
            max_length: int = 128,
            val_tkn_budget: int = 200000
        ): 
        """
        Initialize the dataset for finetuning.
        """
        num_train = self.data_config["num_train"]
        streaming = self.data_config.get("streaming", False)
        downloaded = self.data_config.get("downloaded", False)
        inference = False
        n_train_tkns = self.n_train_tkns

        if downloaded:
            dataset = self.data_config["dirname"].split("/")[-1]
            
        train_dataset = load_data(
            self.tokenizer, 
            dataset, 
            split=self.data_config["train_split_name"], 
            num_train=num_train, 
            max_length=max_length, 
            pad_sequences=pad_sequences, 
            text_field=self.data_config["text_field"], 
            streaming=streaming, 
            packing=self.packing, 
            inference=inference, 
            n_tkns=n_train_tkns, 
            downloaded=downloaded
        )

        val_dataset = load_data(
            self.tokenizer, 
            dataset, 
            split=self.data_config["test_split_name"], 
            max_length=max_length, 
            pad_sequences=pad_sequences, 
            text_field=self.data_config["text_field"], 
            streaming=streaming, 
            packing=self.packing,  
            n_tkns=val_tkn_budget, 
            downloaded=downloaded
        )
            
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        print("train_dataset: ", train_dataset)
        print("val_dataset: ", val_dataset)

    def reverse_tkns(self, sample, text_field):
        rev_ids = sample["input_ids"]
        rev_ids.reverse()
        sample["input_ids"] = rev_ids
        sample[text_field] = self.tokenizer.decode(rev_ids)
        sample["attention_mask"] = [1] * len(rev_ids)
        return sample
    
    def remove_vowels(sample, tokenizer, text_field):
        for s in sample:
            text = s[text_field]
            text = text.replace("a", "").replace("e", "").replace("i", "").replace("o", "").replace("u", "").replace("A", "").replace("E", "").replace("I", "").replace("O", "").replace("U", "")
            s[text_field] = text
            s["input_ids"] = tokenizer.encode(text)
            s["attention_mask"] = [1] * len(s["input_ids"])
        return sample

    def remove_punctuation(sample, tokenizer, text_field):
        # regex to remove all punctuation
        punctuation_regex = r"[^\w\s]"
        for s in sample:
            text = s[text_field]
            text = re.sub(punctuation_regex, "", text)
            s[text_field] = text
            s["input_ids"] = tokenizer.encode(text)
            s["attention_mask"] = [1] * len(s["input_ids"])
        return sample
    
    def remove_stopwords(sample, tokenizer, text_field):
        stopwords = set(stopwords.words("english"))
        for s in sample:
            s[text_field] = " ".join([word for word in s[text_field].split() if word not in stopwords])
            s["input_ids"] = tokenizer.encode(s[text_field])
            s["attention_mask"] = [1] * len(s["input_ids"])
        return sample
    
    def remove_numbers(sample, tokenizer, text_field):
        num_regex = r"\d+"
        for s in sample:
            s[text_field] = re.sub(num_regex, '', s[text_field])
            s["input_ids"] = tokenizer.encode(s[text_field])
            s["attention_mask"] = [1] * len(s["input_ids"])
        return sample   
    
    def remove_special_chars(sample, tokenizer, text_field):
        spl_char_regex = r"[^\w\s]"
        for s in sample:
            s[text_field] = re.sub(spl_char_regex, '', s[text_field])
            s["input_ids"] = tokenizer.encode(s[text_field])
            s["attention_mask"] = [1] * len(s["input_ids"])
        return sample
    
    def remove_white_space(sample, tokenizer, text_field):
        ws_regex = r"\s+"
        for s in sample:
            s[text_field] = re.sub(ws_regex, ' ', s[text_field])
            s["input_ids"] = tokenizer.encode(s[text_field])
            s["attention_mask"] = [1] * len(s["input_ids"])
        return sample
    
    def upper_case(sample, tokenizer, text_field):
        for s in sample:
            s[text_field] = s[text_field].upper()
            s["input_ids"] = tokenizer.encode(s[text_field])
            s["attention_mask"] = [1] * len(s["input_ids"])
        return sample
    
    def lower_case(sample, tokenizer, text_field):
        for s in sample:
            s[text_field] = s[text_field].lower()
            s["input_ids"] = tokenizer.encode(s[text_field])
            s["attention_mask"] = [1] * len(s["input_ids"])
        return sample
            
            
    def perturb_dataset(self, perturbations):
        if perturbations is None or len(perturbations) == 0:
            return
        
        for perturbation in perturbations:
            if perturbation == "reverse_tkns":
                self.train_dataset = self.train_dataset.map(self.reverse_tkns, fn_kwargs={"text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.reverse_tkns, fn_kwargs={"text_field": self.data_config["text_field"]})
            
            if perturbation == "remove_vowels":
                self.train_dataset = self.train_dataset.map(self.remove_vowels, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.remove_vowels, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
            
            if perturbation == "remove_punctuation":
                self.train_dataset = self.train_dataset.map(self.remove_punctuation, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.remove_punctuation, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
            
            if perturbation == "remove_stopwords":
                self.train_dataset = self.train_dataset.map(self.remove_stopwords, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.remove_stopwords, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
            
            if perturbation == "remove_numbers":
                self.train_dataset = self.train_dataset.map(self.remove_numbers, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.remove_numbers, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
            
            if perturbation == "remove_special_chars":
                self.train_dataset = self.train_dataset.map(self.remove_special_chars, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.remove_special_chars, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
            
            if perturbation == "remove_white_space":
                self.train_dataset = self.train_dataset.map(self.remove_white_space, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.remove_white_space, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
            
            if perturbation == "upper_case":
                self.train_dataset = self.train_dataset.map(self.upper_case, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.upper_case, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
            
            if perturbation == "lower_case":
                self.train_dataset = self.train_dataset.map(self.lower_case, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})
                self.val_dataset = self.val_dataset.map(self.lower_case, fn_kwargs={"tokenizer": self.tokenizer, "text_field": self.data_config["text_field"]})

    def finetune_model(self, sft_config_path, field="text", resume_from_checkpoint=False):
        # Set supervised fine-tuning parameters
        with open(sft_config_path, "r") as f:
            sft_config = json.load(f)
        if self.packing:
            sft_config.update({
                "packing": True
            })
        sft_config["max_seq_length"] = self.max_seq_len
        
        
        if self.lora:
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
        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except: 
            print(f"Training failed with resume_from_checkpoint={resume_from_checkpoint}")
            print(f"Setting resume_from_checkpoint={False}")
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
    parser.add_argument("--target_modules", default=None, type=str, help="Target modules for Lora finetuning")
    parser.add_argument("--domain", default="wiki", type=str, help="Domain of the dataset")
    parser.add_argument("--dataset_config", default="configs/dataset_config.json", type=str, help="dataset[:subset]")
    parser.add_argument("--seed", default=1, type=int, help="Seed for reproducibility")
    parser.add_argument("--lora_rank", default=None, type=int, help="Lora rank")
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, help="If this flag is set, Lora finetuning will be done (full parameter finetuning is the default)", default=False)
    parser.add_argument("--num_train", default=None, type=int, help="Number of training samples")
    parser.add_argument("--n_train_tkns", default=None, type=float, help="Training token budget")
    parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, help="If this flag is set, finetuning will resume from the last checkpoint", default=False)
    parser.add_argument("--pad_sequences", action=argparse.BooleanOptionalAction, help="If this flag is set, include sequences shorter than max_len by padding them (by default, this flag is not set, i.e., we only include consider instances with at least max_seq_len tokens)")
    parser.add_argument("--packing", default=False, action="store_true", help="If this flag is set, pack sequences to max_len (pad_sequences will be ignored if this flag is set)")
    parser.add_argument("--stream_size", default=None, type=int, help="Number of training samples to stream if streaming is set to True")
    parser.add_argument("--enable_tkn_budget", default=False, action="store_true", help="This flag ensures that the model will see only a fixed number of (max_seq_len * num_train) tokens during finetuning (i.e. Model is finetuned on max_seq_len sequences and not samples")
    parser.add_argument(
        "--perturbations",
        choices=["reverse_tkns", "remove_vowels", "remove_punctuation", "remove_stopwords", "remove_numbers", "remove_special_chars", "remove_white_space", "upper_case", "lower_case"],
        help="Choose finetuning corpus perturbation option(s) from list [reverse_seq, remove_vowels, remove_punctuation, remove_stopwords, remove_numbers, remove_special_chars, remove_white_space, upper_case, lower_case]",
        default=None,
        nargs="+"
    )


    args = parser.parse_args() 

    base_model = args.base_model
    model_config_path = args.model_config_path
    lora_config_path = args.lora_config_path
    sft_config_path = args.sft_config_path
    lr = args.lr
    num_train = args.num_train
    domain = args.domain
    dataset_config = args.dataset_config
    lora = args.lora
    seed = args.seed
    lora_rank = args.lora_rank
    resume_from_checkpoint = args.resume_from_checkpoint
    pad_sequences = args.pad_sequences
    packing = args.packing
    perturbations = args.perturbations
    n_train_tkns = args.n_train_tkns
    stream_size = args.stream_size
    target_modules = args.target_modules
    # print("Using GPU(s): ", os.environ["CUDA_VISIBLE_DEVICES"])
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
        data_config = datasets[dataset]
        if num_train is not None:
            data_config["num_train"] = num_train
        num_train = data_config["num_train"]
        max_seq_lens = data_config["max_seq_lens"]
        for max_seq_len in max_seq_lens:
            ft = Finetuning(base_model, domain, lora, seed, lr, lora_rank, max_seq_len, packing, perturbations, n_train_tkns, stream_size, target_modules)
            ft.load_configs(data_config, model_config_path, lora_config_path)
            model_save_dir = ft.model_dir
            if os.path.exists(model_save_dir) and "final_model" in os.listdir(model_save_dir):
                print(f"Skipping {dataset} as it is already finetuned")
                continue
            else:
                print(f"Finetuning model: {model_save_dir}")
            
            ft.load_model()
            hf_repo = data_config["hf_repo"]
            ft.init_dataset(hf_repo, pad_sequences=pad_sequences, max_length=max_seq_len)
            # print(f"FT dataset size: {len(ft.train_dataset)}")
            ft.perturb_dataset(perturbations)
            ft.finetune_model(sft_config_path, field=data_config["text_field"], resume_from_checkpoint=resume_from_checkpoint)