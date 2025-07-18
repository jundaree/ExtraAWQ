import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
from pathlib import Path
import argparse

def main(huggingface_repo):
    
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(huggingface_repo, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(huggingface_repo)

    save_path = Path("../dataset") / huggingface_repo.split("/")[-1]
    save_path.mkdir(parents=True, exist_ok=True)
    # Save the model and tokenizer to the specified path
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save OPT model")
    parser.add_argument("--huggingface_repo", type=str, default="facebook/opt-1.3b", help="Hugging Face repository for the model")
    args = parser.parse_args()

    main(args.huggingface_repo)