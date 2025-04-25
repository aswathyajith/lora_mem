# Given a finetuned model, and base model, decompose their difference into SVD components

import torch
from transformers import AutoModelForCausalLM
from src.utils.model import load_model
from collections import OrderedDict
TARGET_MODULES = ["attention.query_key_value.weight"]

def get_svd_components(A: torch.Tensor):
    """
    Get the SVD decomposition of the difference

    Args:
        A: Model component whose SVD components are to be computed
    Returns:
        A Tensor with the SVD components:
            U: The left singular vectors
            S: The singular values
            VT: The right singular vectors
    """
    # Get the SVD decomposition of the difference
    U, S, VT = torch.linalg.svd(A, full_matrices=False)
    return U, S, VT

def svd_decomposition(
        wts: OrderedDict
    ) -> OrderedDict:
    """
    Get the SVD decomposition of the difference

    Args:
        wts: OrderedDict with the the names of the model components as keys and the weights as values
    Returns:
        An OrderedDict with the the names of the model components as keys and their SVD components as values
    """
    svd_components = OrderedDict()
    for name in wts.keys():
        U, S, VT = get_svd_components(wts[name])
        svd_components[name] = (U, S, VT)
    return svd_components

def compute_weight_differences(
    base_model: AutoModelForCausalLM,
    finetuned_model: AutoModelForCausalLM
) -> dict:
    """
    Compute the differences between the weights of two transformer models.
    
    Args:
        model1: First transformer model
        model2: Second transformer model
        
    Returns:
        dict: Dictionary containing weight differences for each model component
    """
    differences = {}

    # Get state dicts
    state_dict1 = base_model.state_dict()
    state_dict2 = finetuned_model.state_dict()

    # Ensure models have the same architecture
    assert set(state_dict1.keys()) == set(state_dict2.keys()), "Models have different architectures"
    
    for name, param1 in state_dict1.items():
        # check if any of the TARGET_MODULES are in the name
        if not any(module in name for module in TARGET_MODULES):
            continue
        param2 = state_dict2[name]
        
        # Compute absolute difference
        diff = param1 - param2
        differences[name] = diff
        
    
    return differences

def compare_update_svd(
        base_model: AutoModelForCausalLM,
        ff_model: AutoModelForCausalLM,
        lf_model: AutoModelForCausalLM
    ) -> tuple[OrderedDict, OrderedDict]:
    """
    Compute the SVD decompositions of the weight updates for the full-finetuned and LoRA models with respect to the base model

    Args:
        base_model: The base model
        ff_model: The full-finetuned model
        lf_model: The LoRA model
    Returns:
        A tuple of two OrderedDicts, with the first element being the SVD decomposition of the difference between the weights of the full-finetuned model and the base model, and the second element being the SVD decomposition of the difference between the weights of the LoRA model and the base model
    """
    
    ff_model_update = compute_weight_differences(ff_model, base_model)
    lf_model_update = compute_weight_differences(lf_model, base_model)
    ff_delta_svd = svd_decomposition(ff_model_update)
    lf_delta_svd = svd_decomposition(lf_model_update)
    return ff_delta_svd, lf_delta_svd

def find_intruder_dims(
        base_model: AutoModelForCausalLM, 
        finetuned_model: AutoModelForCausalLM,
        k: int = 10,
        threshold: float = 0.2
    ):
    """
    Finds the number of intruder dimensions in the finetuned model (source: https://github.com/reeceshuttle/intruder-dimensions/blob/main/find_intruder_dimensions.py)

    Args: 
        base_model: The base model
        finetuned_model: The finetuned model
        k: The number of top values to consider in the base model
        threshold: The threshold for considering a value as an intruder dimension
    Returns:
        The number of intruder dimensions in the finetuned model
    """

    n_intruder_dims = 0
    
    model_params_for_svd = [name for name in base_model.state_dict().keys() if any(module in name for module in TARGET_MODULES)]
    base_model_state_dict = {name: base_model.state_dict()[name] for name in model_params_for_svd}
    finetuned_model_state_dict = {name: finetuned_model.state_dict()[name] for name in model_params_for_svd}
    base_svd_dict = svd_decomposition(base_model_state_dict)
    finetuned_svd_dict = svd_decomposition(finetuned_model_state_dict)
    for name in base_svd_dict.keys():
        if not any(module in name for module in TARGET_MODULES):
            continue
        U_base, S_base, VT_base = base_svd_dict[name]
        U_tuned, S_tuned, VT_tuned = finetuned_svd_dict[name]
        U_dot_prod = torch.abs(torch.einsum('ji,jk->ik', U_tuned, U_base))
        U_tuned_max, _ = U_dot_prod.max(dim=1)
        n_intruder_dims += len(U_tuned_max[:k][U_tuned_max[:k]<threshold])
    return n_intruder_dims

U_tuned_max, _ = svd_0.max(dim=1)

if __name__ == "__main__":
    base_model, tkzr = load_model("EleutherAI/pythia-1.4b", lora_adapter_path = None)
    finetuned_model, tkzr = load_model("models/pythia-1.4b/packing/perturbations/none/legal/us_bills/full-ft/lr_2e-6/n_tkns_2e6/max_seq_len_128/seed_1/final_model", lora_adapter_path = None)
    lora_model, tkzr = load_model("EleutherAI/pythia-1.4b", lora_adapter_path = "models/pythia-1.4b/packing/perturbations/none/legal/us_bills/lora/r_16/lr_2e-4/n_tkns_2e6/max_seq_len_128/seed_1/final_model", merge_and_unload=True)
    print("Intruder dimensions for LoRA:", find_intruder_dims(base_model, lora_model, threshold=0.7))
    print("Intruder dimensions for Full:", find_intruder_dims(base_model, finetuned_model, threshold=0.7))