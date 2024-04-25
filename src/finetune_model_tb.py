import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig,TrainerCallback
from datasets import load_dataset
from peft import LoraConfig, PeftModel
import torch
from trl import SFTTrainer
# import pandas as pd
import argparse
import json
from transformers.integrations import is_tensorboard_available, TensorBoardCallback
from transformers import TrainerCallback
# from tensorflow.keras.callbacks import TensorBoard

def custom_rewrite_logs(d, mode):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if mode == 'eval' and k.startswith(eval_prefix):
            if k[eval_prefix_len:] == 'loss':
                new_d["combined/" + k[eval_prefix_len:]] = v
        elif mode == 'test' and k.startswith(test_prefix):
            if k[test_prefix_len:] == 'loss':
                new_d["combined/" + k[test_prefix_len:]] = v
        elif mode == 'train':
            if k == 'loss':
                new_d["combined/" + k] = v
    return new_d

class CombinedTensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).
    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writers=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                self._SummaryWriter = None
                # try:
                #     from tensorboardX import SummaryWriter

                #     self._SummaryWriter = SummaryWriter
                # except ImportError:
                #     self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writers = tb_writers

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writers = dict(train=self._SummaryWriter(log_dir=os.path.join(log_dir, 'train')),
                                   eval=self._SummaryWriter(log_dir=os.path.join(log_dir, 'eval')))

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writers is None:
            self._init_summary_writer(args, log_dir)

        for k, tbw in self.tb_writers.items():
            tbw.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    tbw.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(tbw, "add_hparams"):
                tbw.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writers is None:
            self._init_summary_writer(args)

        for tbk, tbw in self.tb_writers.items():
            logs_new = custom_rewrite_logs(logs, mode=tbk)
            for k, v in logs_new.items():
                if isinstance(v, (int, float)):
                    tbw.add_scalar(k, v, state.global_step)
                # else:
                #     logger.warning(
                #         "Trainer is attempting to log a value of "
                #         f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                #         "This invocation of Tensorboard's writer.add_scalar() "
                #         "is incorrect so we dropped this attribute."
                #     )
            tbw.flush()

    def on_train_end(self, args, state, control, **kwargs):
        for tbw in self.tb_writers.values():
            tbw.close()
        self.tb_writers = None


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
                callbacks=[CombinedTensorBoardCallback],
                **sft_config
            )

            trainer.remove_callback(TensorBoardCallback)
            trainer.train()
            



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="EleutherAI/pythia-160m")
    parser.add_argument("--output_dir", default="/net/projects/clab/aswathy/projects/lora_mem/models/pythia-160m-lora")
    parser.add_argument("--model_config_path", default="configs/model_configs/pythia-160m.json", type=str)
    parser.add_argument("--sft_config_path", default="configs/model_configs/sft_configs.json", type=str)
    parser.add_argument("--lora_config_path", default="configs/lora_config.json", type=str)
    parser.add_argument("--dataset", default="wikitext:wikitext-2-raw-v1", type=str, help="dataset[:subset]")

    args = parser.parse_args()

    base_model = args.base_model
    model_config_path = args.model_config_path
    lora_config_path = args.lora_config_path
    sft_config_path = args.sft_config_path
    dataset = args.dataset

    ft = Finetuning(base_model)
    ft.load_model_and_configs(model_config_path, lora_config_path)
    ft.load_data(dataset)
    ft.finetune_model(sft_config_path)
    # ft_model.load_model()
