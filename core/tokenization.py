"""
Tokenization outline for week 1 SFT.

Goal:
Take raw GSM8K examples and turn them into model-ready tensors:

    input_ids
    attention_mask
    labels

The important SFT detail:
The model should receive the prompt and answer as input, but the loss should
only be computed on the answer tokens. Prompt tokens in labels should become
-100, because PyTorch cross entropy ignores label value -100 by default.
"""


IGNORE_INDEX = -100

import torch
from transformers import AutoTokenizer


def format_gsm8k_prompt(question):
    """
    Input:
        question: a string from a GSM8K example.

    Output:
        A string prompt that asks the model to solve the math problem.

    What to do:
        1. Decide the exact prompt format you want.
        2. Include the question somewhere in the prompt.
        3. End the prompt at the place where the model should start answering.

    Example shape:
        "Problem: ...\\nSolution:"
    """
    return f'Problem: {question}\\nSolution: '
    


def format_gsm8k_answer(answer):
    """
    Input:
        answer: a string from a GSM8K example.

    Output:
        A cleaned-up answer string that the model should learn to generate.

    What to do:
        1. Strip extra whitespace.
        2. Keep the chain-of-thought / explanation for now.
        3. Later, you can experiment with formatting the final answer more
           carefully, but do not overcomplicate it yet.
    """
    return answer.strip()


def tokenize_sft_example(example, tokenizer, max_length):
    """
    Input:
        example:
            A single GSM8K row. It should have at least:
                example["question"]
                example["answer"]

        tokenizer:
            A HuggingFace tokenizer, probably loaded with:
                AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

        max_length:
            Maximum number of tokens to keep after tokenization.

    Output:
        A dictionary with three keys:
            input_ids:
                List of token IDs for prompt + answer.

            attention_mask:
                List of 1s and 0s telling the model which tokens are real
                versus padding. For a single unpadded example, this is usually
                all 1s.

            labels:
                List of token IDs used as targets for the loss.
                Same length as input_ids.
                Prompt token positions should be IGNORE_INDEX.
                Answer token positions should match input_ids.

    What to do:
        1. Build the prompt string with format_gsm8k_prompt.
        2. Build the answer string with format_gsm8k_answer.
        3. Combine them into one full text:
               prompt + answer + EOS token
        4. Tokenize the prompt by itself.
           You need this length so you know how many labels to mask.
        5. Tokenize the full text.
        6. Copy input_ids into labels.
        7. Replace the prompt part of labels with IGNORE_INDEX.
        8. Return input_ids, attention_mask, and labels.

    Things to watch:
        - Use add_special_tokens=False at first so token counts are easier to
          reason about.
        - If truncation cuts off part of the sequence, make sure labels still
          have exactly the same length as input_ids.
        - If the whole answer gets truncated away, that example is not useful.
          You can ignore this edge case at first, then handle it later.
    """
    fmt_prompt = format_gsm8k_prompt(example['question'])
    fmt_answer = format_gsm8k_answer(example['answer'])
    prompt_tokens = tokenizer(fmt_prompt, add_special_tokens=False, truncation=False)['input_ids']
    answer_tokens = tokenizer(fmt_answer, add_special_tokens=False, truncation=False)['input_ids'] + [tokenizer.eos_token_id]
    prompt_len = min(len(prompt_tokens), max_length)
    all_tokens = (prompt_tokens + answer_tokens)[:max_length]
    attention_mask = [1 for _ in range(len(all_tokens))]
    labels = ([IGNORE_INDEX for _ in range(prompt_len)] + answer_tokens)[:max_length]
    return {
        'input_ids': all_tokens,
        'attention_mask': attention_mask,
        'labels': labels
    }


def tokenize_sft_dataset(dataset, tokenizer, max_length):
    """
    Input:
        dataset:
            A HuggingFace Dataset containing GSM8K examples.

        tokenizer:
            The tokenizer to pass into tokenize_sft_example.

        max_length:
            Maximum sequence length.

    Output:
        A new dataset where each row has tokenized fields:
            input_ids
            attention_mask
            labels

    What to do:
        1. Use dataset.map.
        2. Inside map, call tokenize_sft_example on each example.
        3. Remove original text columns if you want a clean training dataset.

    Note:
        This function is just a convenience wrapper. You can skip it at first
        and call tokenize_sft_example manually on a few rows while debugging.
    """
    return dataset.map(lambda example: tokenize_sft_example(example, tokenizer, max_length))

    


class SFTDataCollator:
    """
    Purpose:
        Convert a list of tokenized examples into a padded batch.

    Why it exists:
        Examples have different lengths. A model batch needs rectangular
        tensors, so shorter examples need padding.

    Constructor input:
        tokenizer:
            The HuggingFace tokenizer. You can use tokenizer.pad for input_ids
            and attention_mask.

        label_pad_token_id:
            The value to use when padding labels. This should be IGNORE_INDEX,
            because padded label positions should not contribute to loss.

    __call__ input:
        features:
            A list of dictionaries, where each dictionary has:
                input_ids
                attention_mask
                labels

    __call__ output:
        A dictionary of PyTorch tensors:
            input_ids:
                Shape [batch_size, padded_seq_len]

            attention_mask:
                Shape [batch_size, padded_seq_len]

            labels:
                Shape [batch_size, padded_seq_len]

    What to do in __call__:
        1. Pad input_ids and attention_mask using tokenizer.pad.
        2. Find the padded sequence length.
        3. Manually pad each labels list to the same length.
        4. Convert labels into a torch.long tensor.
        5. Add labels into the batch dictionary.
        6. Return the batch.
    """

    def __init__(self, tokenizer, label_pad_token_id=IGNORE_INDEX):
        self._tokenizer = tokenizer
        self._label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        seq_len = max(len(example['input_ids']) for example in features)
        batch_size = len(features)
        out_dict = self._tokenizer.pad([
                {
                    'input_ids': feature['input_ids'], 
                    'attention_mask': feature['attention_mask'] 
                } 
                for feature in features
            ],
            padding=True,
            return_tensors='pt'
        )

        # pad labels
        out_dict['labels'] = torch.tensor([feature['labels'] + ([self._label_pad_token_id] * (seq_len - len(feature['labels']))) for feature in features], dtype=torch.long)
        
        # assertion for shape correctness
        for key in out_dict.keys():
            assert out_dict[key].shape == (batch_size, seq_len)
        return out_dict
        


def ensure_pad_token(tokenizer):
    """
    Input:
        tokenizer:
            A HuggingFace tokenizer.

    Output:
        Nothing. This function mutates the tokenizer if needed.

    What to do:
        1. Check whether tokenizer.pad_token is missing.
        2. If it is missing, set it to tokenizer.eos_token.

    Why:
        Decoder-only models often do not define a pad token by default.
        Padding still needs some token ID, and EOS is a common practical choice.
    """
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    from datasets import Dataset

    rows = [
        {
            "question": "What is 2 + 2?",
            "answer": "2 + 2 = 4\n#### 4",
        },
        {
            "question": "If I have 3 apples and buy 5 more, how many apples?",
            "answer": "3 + 5 = 8\n#### 8",
        },
    ]

    dataset = Dataset.from_list(rows)
    tokenized_dataset = tokenize_sft_dataset(dataset, tokenizer, 1024)
    print(f'Tokenized dataset:\n{tokenized_dataset}')
    collator = SFTDataCollator(tokenizer)
    final_tokens = collator(tokenized_dataset)
    print(f'Final tokens:\n{final_tokens}')

