from transformers import AutoModelForCausalLM, AutoTokenizer

def prompt_model(model_name):
    model_name = "EleutherAI/pythia-410m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer
