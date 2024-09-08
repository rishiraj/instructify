# Instructify 📝

Instructify is a Python library designed to convert CSV files or Hugging Face datasets into Hugging Face Dataset objects, specifically formatted for fine-tuning large language models (LLMs). Inspired by the instruction-based dataset approach described in OpenAI's InstructGPT paper ([2203.02155](https://arxiv.org/abs/2203.02155)), this package helps prepare your data for instruction-based tasks using a chat-like format.

## Features ✨
- **CSV or Hugging Face Dataset Support**: Automatically detect whether the input is a CSV file or a Hugging Face dataset.
- **Customizable Message Formatting**: Supports user, assistant, and system messages with flexible column names.
- **Tokenizer Integration**: Automatically integrates with a pre-trained tokenizer to format messages.
- **Custom Templates**: Apply a custom template or use the tokenizer's default chat format.
- **Easy Fine-Tuning Preparation**: Prepares data for instruction tuning, similar to the InstructGPT format.

## Installation 📦
```bash
pip install instructify
```

## Usage 🚀

### CSV Input

```python
import pandas as pd
from instructify import to_train_dataset

# Example custom template
custom_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Example data
data = {
    "input": ["When was the Library of Alexandria burned down?", "What is the capital of France?"],
    "output": ["I-I think that was in 48 BC, b-but I'm not sure.", "The capital of France is Paris."],
    "instruction": ["Bunny is a chatbot that stutters, and acts timid and unsure of its answers.", None]
}

# Convert data to CSV
df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)

# Generate Hugging Face dataset for fine-tuning
train_dataset = to_train_dataset("data.csv", system="instruction", user="input", assistant="output", model="unsloth/Meta-Llama-3.1-8B-Instruct", custom_template=custom_template)

# Inspect the formatted dataset
print(train_dataset["text"])
```

### Hugging Face Dataset Input

```python
from instructify import to_train_dataset

# Example custom template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Using a Hugging Face dataset
train_dataset = to_train_dataset("yahma/alpaca-cleaned", system="instruction", user="input", assistant="output", model="unsloth/Meta-Llama-3.1-8B-Instruct", custom_template=alpaca_prompt)

# Inspect the formatted dataset
print(train_dataset["text"])
```

## Output Example 📄

The function formats CSV files or Hugging Face datasets into a structured template ready for fine-tuning:

| instruction | input | output |
|-------------|-------|--------|
| Bunny is a chatbot that stutters, and acts timid and unsure of its answers. | When was the Library of Alexandria burned down? | I-I think that was in 48 BC, b-but I'm not sure. |
| None        | What is the capital of France? | The capital of France is Paris. |

### Default Output Format

The `train_dataset["text"]` will output the following instruction-style dataset format when using the default tokenizer template:

```txt
[
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBunny is a chatbot that stutters, and acts timid and unsure of its answers.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhen was the Library of Alexandria burned down?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI-I think that was in 48 BC, b-but I'm not sure.<|eot_id|>",
    
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe capital of France is Paris.<|eot_id|>"
]
```

### Custom Template Output

The `train_dataset["text"]` will output the following format when using a custom template:

```txt
[
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nBunny is a chatbot that stutters, and acts timid and unsure of its answers.\n\n### Input:\nWhen was the Library of Alexandria burned down?\n\n### Response:\nI-I think that was in 48 BC, b-but I'm not sure.<|eot_id|>",
    
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n\n\n### Input:\nWhat is the capital of France?\n\n### Response:\nThe capital of France is Paris.<|eot_id|>"
]
```

## Functionality Overview 🔍

### `to_train_dataset`
This function is the core of the library, enabling both CSV and Hugging Face dataset conversion for LLM fine-tuning.

#### Parameters:
- **`data_source`**: Path to the input CSV file or Hugging Face dataset identifier.
- **`system`** *(optional)*: Column name for system messages (e.g., instructions for the model).
- **`user`**: Column name for user messages (default: `'user'`).
- **`assistant`**: Column name for assistant messages (default: `'assistant'`).
- **`model`**: Model name to load the tokenizer from (default: `'unsloth/Meta-Llama-3.1-8B-Instruct'`).
- **`custom_template`** *(optional)*: Custom template for formatting the chat data.

#### Returns:
- **`Dataset`**: A Hugging Face Dataset, ready for LLM fine-tuning.

## License ⚖️
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing 🤝
We welcome contributions! Feel free to open issues or submit pull requests to help improve Instructify.
