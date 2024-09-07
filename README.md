# Instructify 📝

Instructify is a Python library designed to convert CSV files into Hugging Face Datasets, specifically formatted for fine-tuning large language models (LLMs). Inspired by the instruction-based dataset approach described in OpenAI's InstructGPT paper ([2203.02155](https://arxiv.org/abs/2203.02155)), this package helps prepare your data for instruction-based tasks using a chat-like format.

## Features ✨
- **CSV to Hugging Face Dataset**: Convert CSV files into Hugging Face Dataset objects ready for model fine-tuning.
- **Customizable Message Formatting**: Supports user, assistant, and system messages with flexible column names.
- **Tokenizer Integration**: Automatically integrates with a pre-trained tokenizer to format messages.
- **Easy Fine-Tuning Preparation**: Prepares data for instruction tuning, similar to the InstructGPT format.

## Installation 📦
```bash
pip install instructify
```

## Usage 🚀

```python
import pandas as pd
from instructify import to_train_dataset

# Example data
data = {
    "user_prompt": ["When was the Library of Alexandria burned down?", "What is the capital of France?"],
    "assistant_response": ["I-I think that was in 48 BC, b-but I'm not sure.", "The capital of France is Paris."],
    "system_prompt": ["Bunny is a chatbot that stutters, and acts timid and unsure of its answers.", None]
}

# Convert data to CSV
df = pd.DataFrame(data)
df.to_csv("input.csv", index=False)

# Generate Hugging Face dataset for fine-tuning
train_dataset = to_train_dataset("input.csv", system="system_prompt", user="user_prompt", assistant="assistant_response", model="unsloth/Meta-Llama-3.1-8B-Instruct")

# Inspect the formatted dataset
print(train_dataset["text"])
```

## Output Example 📄

The function formats messages in a structured template ready for fine-tuning:

| System Prompt | User Prompt | Assistant Response |
|---------------|--------------|--------------------|
| Bunny is a chatbot that stutters, and acts timid and unsure of its answers. | When was the Library of Alexandria burned down? | I-I think that was in 48 BC, b-but I'm not sure. |
| None          | What is the capital of France?   | The capital of France is Paris. |

The `train_dataset["text"]` will output the following instruction-style dataset format:

```txt
[
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBunny is a chatbot that stutters, and acts timid and unsure of its answers.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhen was the Library of Alexandria burned down?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI-I think that was in 48 BC, b-but I'm not sure.<|eot_id|>",
    
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe capital of France is Paris.<|eot_id|>"
]
```

## Functionality Overview 🔍

### `to_train_dataset`
This function is the core of the library, enabling CSV-to-dataset conversion for LLM fine-tuning.

#### Parameters:
- **`csv_path`**: Path to the input CSV file.
- **`system`** *(optional)*: Column name for system messages (e.g., instructions for the model).
- **`user`**: Column name for user messages (default: `'user'`).
- **`assistant`**: Column name for assistant messages (default: `'assistant'`).
- **`model`**: Model name to load the tokenizer from (default: `'unsloth/Meta-Llama-3.1-8B-Instruct'`).

#### Returns:
- **`Dataset`**: A Hugging Face Dataset, ready for LLM fine-tuning.

## License ⚖️
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing 🤝
We welcome contributions! Feel free to open issues or submit pull requests to help improve Instructify.
