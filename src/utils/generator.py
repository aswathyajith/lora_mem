from src.utils.model import load_model
import torch 
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from datasets import Dataset

class TokenGenerator: 
    def __init__(
            self, 
            model_name: str, 
            lora_adapter_path: str | None = None, 
            top_k: int = 10, 
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ): 
        self.top_k = top_k
        self.model_name = model_name
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.model, self.tokenizer = load_model(
            model_name=self.model_name, 
            lora_adapter_path=self.lora_adapter_path
        )

    @staticmethod
    def inference_step(
        logits: torch.Tensor, 
        input_id: torch.Tensor, 
        device: torch.device, 
        top_k: int = 10
    ) -> list: 
        """
        Get next token probabilities for a given model and input ids
        """
        next_token_ranks = []
        top_k_tokens = torch.tensor([], device=device, dtype=torch.int64)
        top_k_probs = torch.tensor([], device=device, dtype=torch.float32)
        entropies = torch.tensor([], device=device, dtype=torch.float32)
        # Get probabilities of the actual next token at each position
        next_token_probs = []

        # Calculate probability for the actual next token at each position
        for i in range(logits.size(1) - 1):
            next_token_logits = logits[:, i, :]
            next_token_probs_dist = F.softmax(next_token_logits, dim=-1)
            entropy = -torch.sum(next_token_probs_dist * torch.log(next_token_probs_dist))
            actual_next_token_id = input_id[0, i + 1]
            actual_next_token_prob = next_token_probs_dist[0, actual_next_token_id].item()
            next_token_probs.append(actual_next_token_prob)

            # Calculate rank of the actual next token
            _, indices = torch.sort(next_token_probs_dist[0], descending=True)
            actual_next_token_rank = torch.where(indices == actual_next_token_id)[0][0].item() + 1
            next_token_ranks.append(actual_next_token_rank)

            # Get top k predictions
            top_k_token_preds = torch.topk(next_token_probs_dist[0], top_k)
            top_k_token_ids = top_k_token_preds.indices.reshape(1, -1)
            top_k_token_probs = torch.round(top_k_token_preds.values, decimals=3).reshape(1, -1)
            
            top_k_tokens = torch.cat((top_k_tokens, top_k_token_ids))
            top_k_probs = torch.cat((top_k_probs, top_k_token_probs))

            entropies = torch.cat((entropies, torch.tensor([entropy], device=device)))

        return {
            "next_token_probs": next_token_probs, 
            "next_token_ranks": next_token_ranks, 
            "top_k_tokens": top_k_tokens, 
            "top_k_probs": top_k_probs, 
            "entropies": entropies
        }

    def generate_tokens(self, 
        input_ids: torch.Tensor
    ) -> list: 
        """
        Get next token probabilities for a given model and input ids
        """
    
        output_tkns = defaultdict(list)
        device = input_ids.device
        model = self.model
        tokenizer = self.tokenizer

        for instance_num, input_id in enumerate(input_ids):
            
            print(instance_num)

            
            with torch.no_grad():
                outputs = model(input_id)
                logits = outputs.logits

                next_tkn_outputs = TokenGenerator.inference_step(
                    logits=logits, 
                    input_id=input_id, 
                    device=device, 
                    top_k=self.top_k
                )
                
            next_token_probs = next_tkn_outputs["next_token_probs"]
            next_token_ranks = next_tkn_outputs["next_token_ranks"]
            top_k_tokens = next_tkn_outputs["top_k_tokens"]
            top_k_probs = next_tkn_outputs["top_k_probs"]
            entropies = next_tkn_outputs["entropies"]

            # Normalize probabilities for visualization
            next_token_probs = np.array(next_token_probs)
            next_token_ranks = np.array(next_token_ranks)
            top_k_tokens = top_k_tokens.cpu().numpy()
            top_k_probs = top_k_probs.cpu().numpy()
            entropies = entropies.cpu().numpy()
            #norm_probs = (next_token_probs - next_token_probs.min()) / (next_token_probs.max() - next_token_probs.min())
            norm_probs = next_token_probs
            

            # Get curr and next tokens 
            prev_token_ids = [inp_id.item() for inp_id in input_id[:, :-1][0]]
            prev_tokens = tokenizer.convert_ids_to_tokens(prev_token_ids)

            curr_token_ids = [inp_id.item() for inp_id in input_id[:, 1:][0]]
            curr_tokens = tokenizer.convert_ids_to_tokens(curr_token_ids)
            
            in_token_ids = [input_id[:, :-1][0][max(0, i-128):i+1].cpu().numpy() for i in range(len(input_id[:, :-1][0]))]
            uniq_prev_tokens = [len(set(ctx)) for ctx in in_token_ids]
            context_len = [len(ctx) for ctx in in_token_ids]
            in_tkns = tokenizer.batch_decode(in_token_ids)

            instance_nums = [instance_num] * len(prev_tokens)
            
            output_tkns["seq_id"].extend(instance_nums)
            output_tkns["prev_token"].extend(prev_tokens)
            output_tkns["curr_token"].extend(curr_tokens)
            output_tkns["in_tokens"].extend(in_tkns)
            output_tkns["prev_token_id"].extend(prev_token_ids)
            output_tkns["curr_token_id"].extend(curr_token_ids)
            output_tkns["in_token_ids"].extend(in_token_ids)
            output_tkns["curr_token_prob"].extend(next_token_probs)
            output_tkns["curr_token_rank"].extend(np.array(next_token_ranks))
            output_tkns["top_k_pred_tokens"].extend(top_k_tokens)
            output_tkns["top_k_pred_probs"].extend(top_k_probs)
            output_tkns["entropy"].extend(entropies)
            output_tkns["uniq_prev_tokens"].extend(uniq_prev_tokens)
            output_tkns["context_len"].extend(context_len)
            output_tkns["norm_probs"].extend(norm_probs)

        return output_tkns
    
    def get_reshaped_input_ids(
            ds: Dataset, 
            device: str
    ) -> torch.Tensor: 
        """
        Get reshaped input ids for a given model
        """

        input_ids = ds["input_ids"]
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(device)
        input_ids = torch.reshape(input_ids, (input_ids.shape[0], 1, input_ids.shape[1]))
        return input_ids
    
    def iterate_over_ds(
            self, 
            ds: Dataset
    ) -> torch.Tensor: 
        """
        Iterate over a dataset and generate next token outputs at each position
        """

        input_ids = TokenGenerator.get_reshaped_input_ids(ds, device=self.device)
        
        print(input_ids.shape)
        model_outputs = self.generate_tokens(input_ids)
        return model_outputs