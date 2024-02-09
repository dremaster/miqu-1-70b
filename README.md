---
{}
---
# miqu 70b

First model in the potential series.

## Prompt format: Mistral

```
<s> [INST] QUERY_1 [/INST] ANSWER_1</s> [INST] QUERY_2 [/INST] ANSWER_2</s>...
```

Beware that some backends (like llama.cpp) add bos already (by default), so you don't need to prepend it yourself.

## Settings

DO NOT CHANGE ROPE SETTINGS. This model uses high freq base with 32k seen tokens, it should be fine for most tasks.

Only tested with temp 1 and top_p 0.95 with everything else disabled.


This code may work with internet activated on kaggle, not working for me. Please help.

import os
import json
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

mistral_base_path = './mistral_json/'

class MistralForQuestionAnsweringWrapper(nn.Module):
    def __init__(self, model_states_path):
        super().__init__()

#Load the state dictionaries from JSON files
        model_states = ['sample_q2_K.json', 'sample_q4_km.json', 'sample_q5_KM.json']

        model_dict = OrderedDict()

        for filename in model_states:
            with open(os.path.join(model_states_path, filename), 'r') as jf:
                data = json.load(jf)
                state_dict = {k: torch.tensor(v) for k, v in data.items()}
                model_dict.update(state_dict)

        # Initialize the Question Answering model from Hugging Face
        # Set local_files_only=True to avoid fetching the model from the web
        config = AutoConfig.from_pretrained('bert-base-uncased', local_files_only=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased', config=config, local_files_only=True)

        # Override the model parameters with the saved ones
        self.model.load_state_dict(model_dict)

        # Freeze the layers
        for param in self.model.parameters():
            param.requires_grad = False

# Instantiate the custom model
try:
    custom_qa_model = MistralForQuestionAnsweringWrapper(mistral_base_path)
except Exception as e:
    print(f"Error occurred while initializing the model: {e}")
else:
    # Evaluate the model
    custom_qa_model.eval()

    # Example prompt
    query_prompt = "QUERY_1\nMISTRAL\nWhat is the difference between LSTM and GRU?"

    # Obtain the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)

    # Convert the query prompt to input ids
    input_ids = tokenizer(query_prompt, return_tensors='pt', padding=True, truncation=True).input_ids

    # Convert the attention masks
    attention_mask = torch.ones_like(input_ids)

    # Generate predictions
    with torch.no_grad():
        output = custom_qa_model(input_ids, attention_mask)

        # Find top logits for start and end positions
        start_indices = torch.argmax(output["start_logits"], dim=-1)
        end_indices = torch.argmax(output["end_logits"], dim=-1)

        generated_answers = []
        for idx in range(len(start_indices)):
            predicted_start = start_indices[idx].item()
            predicted_end = end_indices[idx].item()
            decoded_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[idx][predicted_start:predicted_end+1]))
            generated_answers.append(decoded_answer)

    print("Generated answers:")
    print("-"*50)
    print(f"Original Query: {query_prompt}\n")
    for idx, ans in enumerate(generated_answers):
        print(f"\nPredicted Start Index: {start_indices[idx]}")
        print(f"Predicted End Index: {end_indices[idx]}\n")
        print(f"Decoded Answer: {ans}")
