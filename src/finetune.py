import os
import yaml
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from data_utils import get_processed_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("Initializing Quantization configurations (Forced Float16 for T4 compatibility)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Enforce stable precision calculations
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Enforce standard FP16 allocation across base model tensors
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = get_processed_dataset(config['training']['dataset'])

    # Configured to prevent library background-casting conflicts
    training_args = SFTConfig(
        output_dir=config['model']['output_dir'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        logging_steps=config['training']['logging_steps'],
        max_steps=config['training']['max_steps'],
        
        # Hardware precision alignment flags
        fp16=False,
        bf16=False, 
        gradient_checkpointing=False, 
        optim="paged_adamw_32bit",   # Use standard 32-bit tracking to prevent scale calculation errors
        
        report_to="none",
        dataset_text_field="text",
        max_length=512
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args
    )

    logger.info("Starting training run...")
    trainer.train()
    
    logger.info("Saving trained LoRA adapters...")
    trainer.model.save_pretrained(config['model']['output_dir'])
    tokenizer.save_pretrained(config['model']['output_dir'])

if __name__ == "__main__":
    run_training()