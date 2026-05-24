import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_prompt(sample):
    """
    Formats raw data structurally matching a standard instruction fine-tuning schema.
    """
    # Using a general QA structure adjustable to your educational dataset
    system_prompt = "You are an expert academic writing assistant. Analyze the user's input and provide structural corrections."
    user_input = sample.get('question', '')
    response = sample.get('answer', '')
    
    formatted_text = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_input}<|end|>\n<|assistant|>\n{response}<|end|>"
    return {"text": formatted_text}

def get_processed_dataset(dataset_name):
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    # Take a small slice for free-tier speed execution
    dataset = dataset.select(range(500))
    
    logger.info("Applying structural educational templates...")
    dataset = dataset.map(format_prompt)
    return dataset