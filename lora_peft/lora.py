from typing import Tuple, List, Dict, Callable
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    TrainingArguments,
    EvalPrediction,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from torch import cuda as torch_cuda, device as torch_device
from torch.backends import mps as torch_mps


def __get_device() -> torch_device:
    if torch_cuda.is_available():
        return torch_device("cuda")
    elif torch_mps.is_available():
        return torch_device("mps")
    return torch_device("cpu")


DEVICE = __get_device()


def load_tokenizer_and_model_for_causal_lm(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)  # Move model to the selected device
    return tokenizer, model


def load_peft_model(
    model: AutoModelForCausalLM,
    config: LoraConfig,
) -> PeftModel:
    model = get_peft_model(model, config)
    model.to(DEVICE)  # Move model to the selected device
    return model


def get_tokenized_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    splits: List[str] = ["train", "validation", "test"],
    dataset_text_key: str = "text",
) -> Dict[str, Dataset]:
    tokenized_datasets = {}
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for split in splits:
        filtered_dataset = dataset[split].filter(lambda x: len(x[dataset_text_key]) > 0)
        tokenized_datasets[split] = filtered_dataset.map(
            lambda x: tokenizer(
                x[dataset_text_key],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE),
            batched=True,
        )
    return tokenized_datasets


def create_lora_config(
    rank: int = 8,
    lora_alpha: int = 32,
    target_modules: List[str] = ["q_proj", "v_proj"],
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )


def print_trainable_parameters(model: PeftModel) -> None:
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def save_pretrained_peft_model(model: PeftModel, save_directory: Path) -> None:
    model.save_pretrained(save_directory=save_directory)


def get_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)


def get_training_args(
    output_dir: str,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.01,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "loss",
    push_to_hub: bool = False,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        push_to_hub=push_to_hub,
    )


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    return {"loss": eval_pred.loss}


def train_model(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    tokenized_datasets: Dict[str, Dataset],
    training_args: TrainingArguments,
    data_collator: DataCollatorWithPadding,
    compute_metrics: Callable[[EvalPrediction], Dict[str, float]],
    do_train: bool = True,
) -> None:
    model.to(DEVICE)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if do_train:
        trainer.train()
    return trainer
