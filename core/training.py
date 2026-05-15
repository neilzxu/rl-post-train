"""
Small training-loop scaffold for Week 1 SFT.

Goal:
Take batches produced by SFTDataCollator and run ordinary causal-LM training:

    forward -> loss -> backward -> optimizer step

The main learning target here is gradient accumulation. That means you can
pretend several small mini-batches are one larger batch by delaying
optimizer.step().
"""


import torch

def move_batch_to_device(batch, device):
    """
    Input:
        batch:
            A dictionary from the dataloader. Expected keys:
                input_ids
                attention_mask
                labels

            Each value should be a torch.Tensor with shape:
                [batch_size, seq_len]

        device:
            Usually "cuda" or "cpu".

    Output:
        A new dictionary where every tensor value has been moved to device.

    What to do:
        1. Loop over batch.items().
        2. Move each tensor with tensor.to(device).
        3. Return the new dictionary.

    Things to watch:
        - Do not mutate the input batch unless you mean to.
        - Later, batches may contain non-tensor metadata. For now, this can
          assume every value is a tensor.
    """
    return {key: v.to(device) for key, v in batch.items()}


def train_one_epoch(model, dataloader, optimizer, device, grad_accum_steps=1):
    """
    Input:
        model:
            A HuggingFace AutoModelForCausalLM.

        dataloader:
            A PyTorch DataLoader yielding dictionaries with:
                input_ids
                attention_mask
                labels

        optimizer:
            A torch optimizer, for example torch.optim.AdamW.

        device:
            Usually "cuda" or "cpu".

        grad_accum_steps:
            Number of mini-batches to accumulate before optimizer.step().
            Example:
                grad_accum_steps=4 means four backward passes, then one
                optimizer update.

    Output:
        Average unscaled loss across mini-batches for the epoch.

    What to do:
        1. Put the model in train mode with model.train().
        2. Clear old gradients with optimizer.zero_grad().
        3. For each batch:
            a. Move it to device.
            b. Call outputs = model(**batch).
            c. Read the scalar loss from outputs.loss.
            d. Save loss.item() for logging before scaling it.
            e. Divide loss by grad_accum_steps.
            f. Call backward() on the scaled loss.
            g. Every grad_accum_steps batches, call optimizer.step() and
               optimizer.zero_grad().
        4. If the epoch ends with leftover accumulated gradients, do one final
           optimizer.step().
        5. Return the average of the unscaled losses.

    Things to watch:
        - HuggingFace causal LM models shift labels internally. Do not shift
          labels yourself.
        - labels should already contain -100 for ignored prompt/pad positions.
        - Track the original loss.item(), not loss.item() after division.
        - If dataloader is empty, decide what you want to return.
    """
    model.train()
    optimizer.zero_grad()
    losses = []
    accum_steps = 0
    for batch in dataloader: 
        batch = move_batch_to_device(batch, device)
        print(batch)
        outputs = model(**batch)
        # is loss scalar here or?
        loss = outputs.loss
        losses.append(loss.item())
        accum_steps += 1
        (loss / grad_accum_steps).backward()
        if accum_steps == grad_accum_steps:
            optimizer.step()
            optimizer.zero_grad()
            accum_steps = 0
    if accum_steps > 0:
        optimizer.step()
        optimizer.zero_grad()
    return torch.mean(torch.tensor(losses))


def run_tiny_train_smoke_test():
    """
    Optional manual test you can fill in after train_one_epoch works.

    Goal:
        Verify the training loop can run end to end on two tiny examples.

    What to do:
        1. Load Qwen/Qwen2.5-1.5B-Instruct tokenizer and model.
        2. Call ensure_pad_token(tokenizer).
        3. Build a tiny Dataset from two GSM8K-like rows.
        4. Tokenize it with tokenize_sft_dataset.
        5. Build SFTDataCollator(tokenizer).
        6. Build a DataLoader.
        7. Create torch.optim.AdamW(model.parameters(), lr=...).
        8. Call train_one_epoch for one epoch.
        9. Print the loss.

    Things to watch:
        - This may be slow or impossible on CPU with the 1.5B model.
        - If using CUDA, move the model to "cuda" before training.
        - Keep the dataset tiny; this is only a smoke test.
    """
    from core.tokenization import ensure_pad_token, tokenize_sft_dataset, SFTDataCollator

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ensure_pad_token(tokenizer)
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
    #print(f'Tokenized dataset:\n{tokenized_dataset}')
    collator = SFTDataCollator(tokenizer)
    dataloader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True, collate_fn=collator)
    model = model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.01)
    loss = train_one_epoch(model, dataloader, optimizer, 'cuda', 1)
    #print(f'Final tokens:\n{final_tokens}')
if __name__  == '__main__':
    run_tiny_train_smoke_test()