
from lora_peft.lora import load_peft_model


def main():
    model = load_peft_model(model_name="gpt2")
    
