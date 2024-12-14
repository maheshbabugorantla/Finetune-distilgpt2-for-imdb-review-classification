# **Apply Lightweight Fine-Tuning to a Foundation Model using PEFT**

## **What is PEFT**

PEFT (Parameter Efficient Fine-Tuning) is a technique that allows fine-tuning of large language models (LLMs) with significantly reduced computational and memory requirements. Instead of updating all parameters in a model during fine-tuning, PEFT focuses on modifying only a small subset of parameters while keeping the majority of the pre-trained model frozen.

### **Key benefits of PEFT**:

- Dramatically reduces memory usage and computational costs
- Maintains model performance comparable to full fine-tuning
- Enables fine-tuning on consumer hardware
- Prevents catastrophic forgetting of pre-trained knowledge

Common PEFT methods:

1. LoRA (Low-Rank Adaptation)
   - Adds trainable rank decomposition matrices to transformer layers
   - Only trains these small adapter matrices while keeping base model frozen
   - Typically uses ranks between 4-32

2. Prefix Tuning
   - Prepends trainable continuous prompts to inputs
   - Only updates these prefix tokens during training
   - Particularly effective for encoder-decoder models

3. Prompt Tuning
   - Similar to prefix tuning but adds prompts only to the input
   - Even more parameter efficient than prefix tuning
   - Works well for task-specific adaptations

PEFT can be selectively applied to specific layers (encoder/decoder) based on the task:

- For classification: Focus on encoder layers
- For generation: Prioritize decoder layers
- For translation: Apply to both encoder and decoder

This targeted approach further reduces parameters while maintaining task performance.

## **Project Summary**

Use `PyTorch` + `HuggingFace` (`peft` library) training and inference process. Specifically, you will:

1. Load a pre-trained (`gpt2`) model and evaluate its performance
2. Perform parameter-efficient fine tuning using the pre-trained model
3. Perform inference using the fine-tuned model and compare its performance to the original model

## **Results**

Please check [Model Evaluation Notebook](./Model_Evaluation.ipynb) for all the details and explanations on the results.

Code is all written in [sequence_classification.py](./lora_peft/sequence_classification.py) file
