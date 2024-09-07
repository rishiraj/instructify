import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

def to_train_dataset(csv_path, system=None, user='user', assistant='assistant', model='unsloth/Meta-Llama-3.1-8B-Instruct', custom_template=None):
    """
    Converts a CSV to a Hugging Face Dataset, formatted for LLM fine-tuning.
    
    Args:
        csv_path (str): Path to the CSV file.
        system (str): Column name for system messages (optional).
        user (str): Column name for user messages.
        assistant (str): Column name for assistant messages.
        model (str): Model name for loading the tokenizer (default: unsloth/Meta-Llama-3.1-8B-Instruct).
        custom_template (str): Optional custom template for formatting the chat data.
        
    Returns:
        Dataset: Hugging Face Dataset ready for fine-tuning.
    """
    
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Initialize the tokenizer from the model
    tokenizer = AutoTokenizer.from_pretrained(model)
    EOS_TOKEN = tokenizer.eos_token  # End of Sequence token
    
    # Create a "messages" column by combining system, user, and assistant columns
    def create_messages(row):
        messages = []
        if system and pd.notnull(row[system]):
            messages.append({'content': row[system], 'role': 'system'})
        messages.append({'content': row[user], 'role': 'user'})
        messages.append({'content': row[assistant], 'role': 'assistant'})
        return messages
    
    df['messages'] = df.apply(create_messages, axis=1)
    
    # Apply either the custom template or the default tokenizer template
    def apply_template(row):
        if custom_template:
            # If custom template is provided, use it
            instruction = row[system] if system and pd.notnull(row[system]) else ""
            input = row[user]
            output = row[assistant]
            return custom_template.format(instruction, input, output) + EOS_TOKEN
        else:
            # Use tokenizer's chat template if custom template is not provided
            chat = row['messages']
            return tokenizer.apply_chat_template(chat, tokenize=False)
    
    df['text'] = df.apply(apply_template, axis=1)
    
    # Convert DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df)
    
    return hf_dataset

# Example usage
if __name__ == "__main__":
    # Example call to the function with a custom template
    custom_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    data = {
        "input": ["When was the Library of Alexandria burned down?", "What is the capital of France?"],
        "output": ["I-I think that was in 48 BC, b-but I'm not sure.", "The capital of France is Paris."],
        "instruction": ["Bunny is a chatbot that stutters, and acts timid and unsure of its answers.", None]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)
    
    train_dataset = to_train_dataset("data.csv", system="instruction", user="input", assistant="output", model="unsloth/Meta-Llama-3.1-8B-Instruct", custom_template=custom_template)
    print(train_dataset["text"])
