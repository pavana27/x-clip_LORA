import torch
import numpy as np
import transformers
from transformers import AutoProcessor, XCLIPVisionModel
import pandas as pd
from peft import LoraConfig, get_peft_model

from datasets import load_dataset
dataset = load_dataset("pavitemple/Xclip-finetuning")
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
#fucntion to print number of trainable parameter using LORA
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}" 
        )
#If only targeting attention blocks of the model
target_modules = ["q_proj", "v_proj"]

#If targeting all linear layers
#target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

lora_config = LoraConfig(
r=16,
target_modules = target_modules,
lora_alpha=8,
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM")

#prints number of trainable parameters
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

trainer = transformers.Trainer(
model = model,
train_dataset=dataset['train'],
eval_dataset = dataset['validation'],
args = transformers.Seq2SeqTrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        logging_steps=1,
        remove_unused_columns=False,
        output_dir="outputs",
        )
)
model.config.use_cache = False #silence the warning. please re-enable for inference!
# Initiate the training process
#with mlflow.start_run(run_name= "run_name_of_choice")
trainer.train()
