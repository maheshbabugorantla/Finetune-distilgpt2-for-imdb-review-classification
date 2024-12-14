from typing import Tuple, Dict, List, Callable
from argparse import ArgumentParser
from json import loads as json_loads

import numpy as np
from pandas import DataFrame
from torch import cuda as torch_cuda, device as torch_device
from torch.backends import mps as torch_mps
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
    AutoPeftModelForSequenceClassification
)


def __get_device() -> torch_device:
    if torch_cuda.is_available():
        return torch_device("cuda")
    elif torch_mps.is_available():
        return torch_device("mps")
    return torch_device("cpu")


DEVICE = __get_device()


def load_tokenizer_and_model(
    model_name: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    num_labels: int,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = True

    return tokenizer, model


def get_tokenized_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    splits: List[str] = ["train", "validation", "test"],
    text_column_name: str = "text",
) -> Dict[str, Dataset]:
    tokenized_datasets: Dict[str, Dataset] = {}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for split in splits:
        filtered_dataset = dataset[split].filter(lambda x: len(x[text_column_name]) > 0)
        tokenized_datasets[split] = filtered_dataset.map(
            lambda x: tokenizer(
                x[text_column_name],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ),
            batched=True,
        )
    return tokenized_datasets


def get_peft_config(
    r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        inference_mode=False,
    )


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}


def get_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)


def get_training_args(
    output_dir: str = "./results",
    evaluation_strategy: str = "epoch",
    num_train_epochs: int = 256,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    weight_decay: float = 0.01,
    learning_rate: float = 5e-5,
    lr_scheduler_type: str = "cosine",
    logging_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    push_to_hub: bool = False,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = True,
    max_grad_norm: float = 1.0,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        logging_strategy=logging_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        push_to_hub=push_to_hub,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        fp16=DEVICE.type == "cuda",
    )


def get_trainer(
    model: AutoModelForSequenceClassification,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )


def get_peft_model_for_text_classification(
    model: AutoModelForSequenceClassification,
    peft_config: LoraConfig,
) -> PeftModel:
    peft_model = get_peft_model(model, peft_config)
    for param in peft_model.parameters():
        param.requires_grad = True
    return peft_model


def fine_tune_model_using_peft(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
    data_collator: DataCollatorWithPadding,
    compute_metrics: Callable[[EvalPrediction], Dict[str, float]],
    save_model: bool = True,
) -> None:
    model.to(DEVICE)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model and tokenizer
    if save_model:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


def get_predictions(
    test_dataset: Dataset,
    trainer: Trainer,
    text_column_name: str = "text",
) -> DataFrame:
    results = trainer.predict(test_dataset)
    df = DataFrame(
        {
            "text": [item[text_column_name] for item in test_dataset],
            "predictions": results.predictions.argmax(axis=1),
            "labels": results.label_ids,
        }
    )
    return df


def load_peft_model_from_checkpoint(
    output_dir: str,
    id_to_label: Dict[int, str],
    label_to_id: Dict[str, int],
) -> AutoPeftModelForSequenceClassification:
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        output_dir,
        id2label=id_to_label,
        label2id=label_to_id,
        num_labels=len(id_to_label)
    )
    model.to(DEVICE)
    return model


def load_tokenizer_from_checkpoint(output_dir: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model_trainer(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    tokenized_datasets: Dict[str, Dataset],
    training_args: TrainingArguments,
) -> Trainer:
    # Move model to underlying available gpu
    model.to(DEVICE)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer


def load_trainer_from_checkpoint(
    output_dir: str,
    tokenized_datasets: Dict[str, Dataset],
    id_to_label: Dict[int, str],
    label_to_id: Dict[str, int]
) -> Tuple[Dict[str, Dataset], AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer]:
    model = load_peft_model_from_checkpoint(
        output_dir, id_to_label=id_to_label, label_to_id=label_to_id
    )
    tokenizer = load_tokenizer_from_checkpoint(output_dir)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    training_args = get_training_args(output_dir)
    trainer = get_model_trainer(
        tokenizer, model, tokenized_datasets, training_args
    )
    return tokenizer, model, training_args, trainer


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", required=True)
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--id2label", type=str, required=True)
    parser.add_argument("--label2id", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--text_column_name", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    id_to_label = {int(k): v for k, v in json_loads(args.id2label).items()}
    label_to_id = json_loads(args.label2id)

    tokenizer, model = load_tokenizer_and_model(
        args.model_name, id_to_label, label_to_id, args.num_labels
    )
    dataset = load_dataset(args.dataset_name)
    tokenized_dataset = get_tokenized_dataset(
        dataset, tokenizer, args.splits, args.text_column_name
    )

    peft_config = get_peft_config(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    peft_model = get_peft_model_for_text_classification(model, peft_config)
    data_collator = get_data_collator(tokenizer)
    training_args = get_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
    )
    fine_tune_model_using_peft(
        peft_model,
        tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        training_args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        save_model=True,
    )

    print(f"Saved LoRA PEFT model: {args.model_name} -> {args.output_dir}")

    # Load the fine-tuned model
    # model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    # model.to(DEVICE)

    # # Evaluate the model
    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for batch in tqdm(tokenized_dataset["test"]):
    #         inputs = {k: v.unsqueeze(0) for k, v in batch.items() if k != "label"}
    #         labels = batch["label"].unsqueeze(0)
    #         outputs = model(**inputs)
    #         predictions = torch.argmax(outputs.logits, dim=-1)
    #         correct += (predictions == labels).sum().item()
    #         total += labels.size(0)

    # accuracy = correct / total
    # print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
