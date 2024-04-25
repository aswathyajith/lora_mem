from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
import torch
import pandas as pd
import argparse
import os

class Prompter:
    def __init__(self, base_model_name, peft_weights_path, model_type='pretrained'):
        device_map="auto"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        self.tokenizer = tokenizer

        if model_type ==  "lora":
            model = PeftModel.from_pretrained(model, peft_weights_path)
            model = model.merge_and_unload()

        self.model = model

    def prep_data(self, dataset, subset=None, max_length=128): 
        # prep data 

        def tokenize(sample):
            return self.tokenizer(sample["text"], padding='max_length', truncation=True, max_length=max_length)

        if subset is not None:
            ds = load_dataset(dataset, subset)
        else: 
            ds = load_dataset(dataset)
        train_dataset = ds["train"]
        train_dataset = train_dataset.map(tokenize, batched=True)
        train_dataset.set_format("pt", columns=["input_ids"], output_all_columns=True)
        val_dataset = ds["validation"]
        val_dataset = val_dataset.map(tokenize, batched=True)
        val_dataset.set_format("pt", columns=["input_ids"], output_all_columns=True)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def prompt_model(self, batch, prompt_context_len, print_seqs=False, tokenized=False):

        if not tokenized: 
            full_inp_ids = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda").input_ids # encode batch of input seq
        else: 
            full_inp_ids = batch["input_ids"].to("cuda")
        # print(full_inp_ids)
        input_ids = full_inp_ids[:,:prompt_context_len] # get the input tokens that we want to prompt model with 
        inp_prompt = self.tokenizer.batch_decode(input_ids)
        model_comp_ids = self.model.generate(input_ids, max_length=128, early_stopping=True, do_sample=False)
        model_comp = self.tokenizer.batch_decode(model_comp_ids)

        if print_seqs:
            print("TRAIN SEQS:", batch)
            print("INPUT PROMPTS:", inp_prompt)
            print("INPUT PROMPTS + MODEL COMPLETIONS:", model_comp)
        return batch, inp_prompt, model_comp, full_inp_ids, model_comp_ids

    def select_samples_with_len(self, split="train", prompt_context_len=25): 
        '''Get a subset of the dataset with a minimum context length'''

        ds_split = self.train_dataset if split=="train" else self.val_dataset
        
        # Filter for samples that have minimum length 
        self.prompt_ds = ds_split.filter(lambda sample: sample['input_ids'][prompt_context_len-1] != self.tokenizer.eos_token_id)
        
    def prompt_with_dataset(self, output_file_path, split="train", prompt_context_len=25): 
        self.select_samples_with_len(split=split, prompt_context_len=prompt_context_len)
        df = pd.DataFrame(columns=[f"orig_seq", "inp_prompt", "model_comp"])
        
        # sample_ds = self.prompt_ds.select(range(16))

        save_tensor = []
        for batch in self.prompt_ds.iter(batch_size=16):
            _, inp_prompt, model_comp, orig_seq_ids, model_comp_ids = self.prompt_model(batch, prompt_context_len, tokenized=True)
            # SAve tensors 
            batch_tensor = torch.stack((orig_seq_ids, model_comp_ids), dim=1)
            save_tensor.append(batch_tensor)
            # batch_output_dict = pd.DataFrame({
            #     "orig_seq": batch['text'],
            #     "inp_prompt": inp_prompt,
            #     "model_comp": model_comp
            # })

            # df = pd.concat([df, batch_output_dict], ignore_index=True)
            

        # save model completions
        if not os.path.exists(output_dir_path): 
            os.makedirs(output_dir_path)

        # df.to_csv(os.path.join(output_dir_path, "inp_out_txt.csv"), index=False)
        save_tensor = torch.cat(save_tensor)
        tensor_save_file = os.path.join(output_dir_path, "inp_out_tensors.pt")
        torch.save(save_tensor, tensor_save_file)
        
    def measure_mem(self, model_comp_file, context_len): 
        def get_first_div_pos(a, b): 
            # gets the index of the first pos at which two tensors diverge
            disagreement_mask = a != b
            return torch.argmax(disagreement_mask, dim=1) - context_len

        # Load model completions from disk 
        model_comps = pd.read_csv(model_comp_file)
        model_comps=model_comps.head(1)

        # Tokenize 'orig_seq' and 'model_comp' columns of the dataframe 
        orig_seq_ids = self.tokenizer(model_comps['orig_seq'], padding=True, truncation=True, return_tensors="pt").to("cuda").input_ids
        model_comp_ids = self.tokenizer(model_comps['model_comp'], padding=True, truncation=True, return_tensors="pt").to("cuda").input_ids
        
        return get_first_div_pos(orig_seq_ids, model_comp_ids)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_model_name", default="models/full-ft/pythia-160m/final_model", type=str)
    parser.add_argument("--peft_weights_path", default="models/lora/pythia-160m/final_model", type=str)
    parser.add_argument("--output_dir_path", default="results/pythia-160m/full-ft/epoch_20/train_set_completions/len_25", type=str)
    parser.add_argument("--prompt_context_len", default=25, type=int)
    parser.add_argument("--predefined_context_lens", action=argparse.BooleanOptionalAction)
    parser.add_argument("--data_split", default="train", type=str, help="Dataset split (train, validation, test)")
    parser.add_argument("--model_type", default="full-ft",type=str, help="one of (pretrained, lora, full-ft)")
    
    args = parser.parse_args()

    peft_weights_path = args.peft_weights_path
    base_model_name = args.base_model_name
    output_dir_path = args.output_dir_path
    prompt_context_len = args.prompt_context_len
    split = args.data_split
    model_type = args.model_type
    predefined_context_lens = args.predefined_context_lens

    prompter = Prompter(base_model_name, peft_weights_path, model_type)
    prompter.prep_data("wikitext", "wikitext-2-raw-v1", 128)

    batch = [
        "As with previous Valkyira Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through unvoiced text . The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked . The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game . There are also love simulation elements related to the game 's two main heroines , although they take a very minor role .", 
        "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n .",
        "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the \" Nameless \" , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit \" Calamaty Raven \" .",
        "It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 ."
    ]

    # prompter.prompt_model(batch, prompt_context_len=25, print_seqs=True)

    if predefined_context_lens:
        prompt_context_lens = [25, 50, 75]
        for c_len in prompt_context_lens:

            prompter.prompt_with_dataset(output_dir_path, split=split, prompt_context_len=c_len)
    else: 
        prompter.prompt_with_dataset(output_dir_path, split=split, prompt_context_len=prompt_context_len)

