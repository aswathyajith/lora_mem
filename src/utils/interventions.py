# Given a finetuned model, and base model, decompose their difference into SVD components

import torch
from transformers import AutoModelForCausalLM
from src.utils.model import load_model
from collections import OrderedDict

TARGET_MODULES = {"attn_only": ["query_key_value"]}


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
    U, S, VT = torch.linalg.svd(A.float(), full_matrices=False)
    return U, S, VT


def svd_decomposition(
    wts: OrderedDict, target_modules: list[str], verbose: bool = False
) -> OrderedDict:
    """
    Get the SVD decomposition of wts passed in for the target modules

    Args:
        wts: OrderedDict with the the names of the model components as keys and the weights as values
    Returns:
        An OrderedDict with the the names of the model components as keys and their SVD components as values
    """
    svd_components = OrderedDict()
    for name in wts.keys():
        if len(wts[name].shape) == 1:
            continue
        if not any(module in name for module in target_modules):
            continue
        if verbose:
            print("Computing SVD for", name)
        U, S, VT = get_svd_components(wts[name])
        svd_components[name] = (U, S, VT)
    return svd_components


def compute_weight_differences(
    base_model: AutoModelForCausalLM, finetuned_model: AutoModelForCausalLM
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
    state_dict1: OrderedDict = base_model.state_dict()
    state_dict2: OrderedDict = finetuned_model.state_dict()

    # Ensure models have the same architecture
    assert set(state_dict1.keys()) == set(
        state_dict2.keys()
    ), "Models have different architectures"

    for name, param1 in state_dict1.items():
        param2 = state_dict2[name]

        # Compute absolute difference
        diff = param1 - param2
        differences[name] = diff

    return differences


def truncate_and_merge_svc(svd_state_dict: OrderedDict, r: int) -> OrderedDict:
    """
    Get the low-rank approximation of the SVD components
    """

    low_rank_approx = OrderedDict()
    for name, (U, S, VT) in svd_state_dict.items():
        low_rank_approx[name] = U[:, :r] @ torch.diag(S[:r]) @ VT[:r, :]
    return low_rank_approx


def get_all_linear_modules(model: AutoModelForCausalLM) -> list[str]:
    """
    Get all linear modules in the model
    """
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]


def get_full_model_with_low_rank_update(
    base_model: AutoModelForCausalLM,
    full_model: AutoModelForCausalLM,
    target_module_key: str,
    r: int,
) -> AutoModelForCausalLM:
    """
    Args:
        base_model: The base model
        full_model: The full-finetuned model
    Returns:
        A modified full-finetuned model whose task vector induced by full finetuning is approximated by a low-rank SVD approximation

    This function computes the task vector induced by full finetuning, obtains the SVD components of the parameters matching those in TARGET_MODULES[module_names] and truncates them to rank r. It then merges this low-rank approximation of the task vector with the base model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    full_model.to(device)

    # Get target modules from TARGET_MODULES
    if target_module_key == "all-linear":
        target_modules = get_all_linear_modules(base_model)
    else:
        target_modules = TARGET_MODULES.get(target_module_key, [])

    # Get SVD of the full model's task vector
    full_model_update = compute_weight_differences(full_model, base_model)
    full_delta_svc = svd_decomposition(full_model_update, target_modules)
    full_delta_low_rank_approx = truncate_and_merge_svc(full_delta_svc, r)

    # Create a copy of the base model's state dict and
    # update it by adding the low-rank version of the full model task vector
    # to target_modules parameters
    base_state_dict = OrderedDict(base_model.state_dict())
    new_state_dict = OrderedDict({})
    for name, param in base_state_dict.items():
        # clone to prevent modifying the base model state dict (shallow copy is not enough)
        new_state_dict[name] = param.clone()
        if name in full_delta_low_rank_approx:
            new_state_dict[name] += full_delta_low_rank_approx[name]

    # Create a model with the same config as the base model
    # and load the new state dict
    updated_model = AutoModelForCausalLM.from_config(base_model.config)
    updated_model.to(base_model.device)
    updated_model.load_state_dict(new_state_dict)

    return updated_model


def are_models_equal(
    model1: AutoModelForCausalLM, model2: AutoModelForCausalLM
) -> bool:
    """
    Check if two models are equal
    """
    for name, param in model1.state_dict().items():
        if not torch.allclose(param, model2.state_dict()[name]):
            return False
    return True


def find_intruder_dims(
    base_model: AutoModelForCausalLM,
    finetuned_model: AutoModelForCausalLM,
    k: int = 10,
    threshold: float = 0.2,
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

    model_params_for_svd = [
        name
        for name in base_model.state_dict().keys()
        if any(module in name for module in TARGET_MODULES)
    ]
    base_model_state_dict = {
        name: base_model.state_dict()[name] for name in model_params_for_svd
    }
    finetuned_model_state_dict = {
        name: finetuned_model.state_dict()[name] for name in model_params_for_svd
    }
    base_svd_dict = svd_decomposition(base_model_state_dict)
    finetuned_svd_dict = svd_decomposition(finetuned_model_state_dict)
    total_dims = 0
    for name in base_svd_dict.keys():
        if not any(module in name for module in TARGET_MODULES):
            continue
        U_base, S_base, VT_base = base_svd_dict[name]
        U_tuned, S_tuned, VT_tuned = finetuned_svd_dict[name]
        U_dot_prod = torch.abs(torch.einsum("ji,jk->ik", U_tuned, U_base))
        U_tuned_max, _ = U_dot_prod.max(dim=1)
        n_intruder_dims += len(U_tuned_max[:k][U_tuned_max[:k] < threshold])
        total_dims += k  # at most k intruder dimensions
    return n_intruder_dims, total_dims


if __name__ == "__main__":
    base_model, tkzr = load_model("EleutherAI/pythia-1.4b", lora_adapter_path=None)
    full_model, tkzr = load_model(
        "models/pythia-1.4b/packing/perturbations/none/legal/us_bills/full-ft/lr_2e-6/n_tkns_2e6/max_seq_len_256/seed_1/final_model",
        lora_adapter_path=None,
    )
    # lora_model, tkzr = load_model("EleutherAI/pythia-1.4b", lora_adapter_path = "models/pythia-1.4b/packing/perturbations/none/legal/us_bills/lora/all-linear/r_16/lr_2e-4/n_tkns_2e6/max_seq_len_256/seed_1/final_model", merge_and_unload=True)

    # Get low-rank svd approximation
    r = 16
    full_model_low_rank = get_full_model_with_low_rank_update(
        base_model, full_model, ["attn_only"], r
    )
    if are_models_equal(full_model_low_rank, full_model):
        print("Full model and low-rank full model are equal")
    else:
        print("Full model and low-rank full model are not equal")
    if are_models_equal(full_model_low_rank, base_model):
        print("Low-rank full model and base model are equal")
    else:
        print("Low-rank full model and base model are not equal")

    key = "gpt_neox.layers.0.attention.query_key_value.weight"
    print(base_model.state_dict()[key])
    print(full_model.state_dict()[key])
    print(full_model_low_rank.state_dict()[key])
