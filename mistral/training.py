import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

import sentencepiece # Not used directly, but needed for the tokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd
import gc

os.environ['WANDB_DISABLED'] = 'true' 

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3" # no need to change this
NEW_MODEL = "genre-mistral-best" # Name for the output mode
EPOCHS = 2 # hyperparameters we tuned
BATCH_SIZE = 16
NUM_PROMPTS = -1 # Set to -1 to use the whole dataset


prompts = []
completions = []

# Load the dataset from the test file "genredataset/train_data.txt"
with open("./genredataset/train_data.txt", "r") as f:
    # Append data to the prompts from the file
    for line in f.readlines():
        data = line.split(" ::: ")
        prompts.append("Title: " + data[1] + "\nDescription: " + data[3])
        completions.append(data[2])

# Use this to limit training dataset size
# print(len(prompts))
prompts = prompts[:NUM_PROMPTS]
completions = completions[:NUM_PROMPTS]

# Load the model - you will probably want a GPU to run this.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Get tokenizer and set padding
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  
tokenizer.padding_side = "right"


# Use the tokenizer to generate the chat format:
dataset_list = []
for prompt_num in range(len(prompts)):
    messages = [
        {
            "role": "system",
            "content": "You are a classification assistant. Your job is to provide genres for movies based on their titles and descriptions."
            + "The user will provide you with a title and a description, and you should simply provide a genre that fits the movie. ONLY respond with the genre."
        },
        {"role": "user", "content": prompts[prompt_num]},
        {"role": "assistant", "content": completions[prompt_num]},
    ]

    dataset_list.append(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    )

# Now create a hugging face dataset
pandas_dataset = pd.DataFrame(dataset_list)
local_dataset = Dataset.from_pandas(pandas_dataset)


# Setup Lora config. These are standard hyperparameters, which I have found effective in previous experiments
peft_params = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "k_proj",
        "gate_proj",
        "v_proj",
        "up_proj",
        "q_proj",
        "o_proj",
        "down_proj",
    ],
)

# Set training parameters
training_params = TrainingArguments(
    output_dir="./ckpt/" + NEW_MODEL,  # Used for checkpointing
    num_train_epochs=EPOCHS,  # Number of epochs to train for
    per_device_train_batch_size=BATCH_SIZE,  # These two work together. 16 * 2 = 32
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,  # This is necessary to make it computable with 16 GB of VRAM. Then it only takes 6 GB
    optim="paged_adamw_32bit",
    save_steps=1000, # Save a checkpoint every 1000 steps, in case an error occurs during training
    logging_steps=200,
    learning_rate=2e-4, # typical value
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03, # related to learning rate scheduler
    group_by_length=True,
    warmup_steps=100,
    lr_scheduler_type="cosine", # learning rate scheduler adjusts the learning rate over time
)


# Do the training using SFTTrainer
# These parameters are mostly self explanatory
trainer = SFTTrainer(
    model=model,
    train_dataset=local_dataset,
    peft_config=peft_params,
    dataset_text_field="0", # The name of the field with the prompts in it is '0'
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Run and save the results.
trainer.train()
print("training complete, saving chkpts")

# Save final LoRA adapter checkpoint
trainer.model.save_pretrained("./final_checkpoints/" + NEW_MODEL)
tokenizer.save_pretrained("./final_checkpoints/" + NEW_MODEL)
print("saved the checkpoint data")

# Flush memory
del (
    trainer,
    model,
)  # ref_model
gc.collect()
torch.cuda.empty_cache()

# Reload model to combine it with the learned LoRA adapter
reload_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Merge base model with the adapter
model = PeftModel.from_pretrained(reload_model, "./final_checkpoints/" + NEW_MODEL)
model = model.merge_and_unload()

# Save model and tokenizer
model.save_pretrained("./models/" + NEW_MODEL)
tokenizer.save_pretrained("./models/" + NEW_MODEL)
print("saved the model and tokenizer")
