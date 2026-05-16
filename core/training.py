"""
Small training-loop utilities for Week 1 SFT.

Goal:
Take batches produced by SFTDataCollator and run ordinary causal-LM training:

    forward -> loss -> backward -> optimizer step

The main learning target here is gradient accumulation. That means you can
pretend several small mini-batches are one larger batch by delaying
optimizer.step().
"""


from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainConfig:
    """
    Small bag of training settings.

    Keep this intentionally boring. The point is not to build a framework; the
    point is to keep the knobs for the smoke tests and SFT run in one place.

    Fields:
        num_epochs:
            Number of passes over the dataloader.

        grad_accum_steps:
            Number of mini-batches to accumulate before each optimizer update.

        log_every_steps:
            Log every N optimizer updates. Use optimizer steps, not micro-batches.

        checkpoint_every_steps:
            Save every N optimizer updates. Set to None to disable step-based
            checkpointing.

        output_dir:
            Directory where checkpoints should be written.

        use_wandb:
            Whether to call wandb.log from the training loop.
    """

    num_epochs: int = 1
    grad_accum_steps: int = 1
    log_every_steps: int = 10
    checkpoint_every_steps: int | None = None
    output_dir: str = "checkpoints"
    use_wandb: bool = False


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


def init_wandb_run(project, run_name, config):
    """
    Input:
        project:
            wandb project name, for example "rl-from-scratch".

        run_name:
            Human-readable run name, for example "sft-8-example-overfit".

        config:
            A dictionary or dataclass with hyperparameters to store in wandb.

    Output:
        The wandb run object.

    What to do:
        1. Import wandb inside this function so the rest of the file can run
           even when wandb is not being used.
        2. If config is a dataclass, convert it to a plain dictionary.
        3. Call wandb.init(project=..., name=..., config=...).
        4. Return the run.

    Things to watch:
        - This function should be called once at the start of a script.
        - Do not call wandb.init inside every epoch or every batch.
    """
    pass


def log_train_metrics(metrics, global_step, use_wandb=False):
    """
    Input:
        metrics:
            Dictionary of scalar metrics, for example:
                {
                    "train/loss": 2.31,
                    "train/lr": 2e-5,
                    "train/epoch": 0,
                }

        global_step:
            Number of optimizer updates completed so far.

        use_wandb:
            Whether to send metrics to wandb.

    Output:
        Nothing.

    What to do:
        1. Print a compact human-readable message for local debugging.
        2. If use_wandb is True, import wandb and call:
               wandb.log(metrics, step=global_step)

    Things to watch:
        - global_step should count optimizer steps, not mini-batches.
        - Keep metrics scalar. wandb can log richer objects, but save that
          for later.
    """
    pass


def save_checkpoint(model, optimizer, checkpoint_dir, epoch, global_step):
    """
    Input:
        model:
            The model being trained.

        optimizer:
            The optimizer whose state should be saved.

        checkpoint_dir:
            Directory for this checkpoint, for example:
                "checkpoints/step-000100"

        epoch:
            Current epoch index.

        global_step:
            Number of optimizer updates completed so far.

    Output:
        Nothing.

    What to do:
        1. Create checkpoint_dir if it does not exist.
        2. Save model weights.
           Standard HuggingFace option:
               model.save_pretrained(checkpoint_dir)
        3. Save optimizer/training state with torch.save, for example:
               {
                   "optimizer": optimizer.state_dict(),
                   "epoch": epoch,
                   "global_step": global_step,
               }
        4. Put that state in something like:
               checkpoint_dir / "training_state.pt"

    Things to watch:
        - model.save_pretrained saves model weights/config, not optimizer state.
        - torch.save is fine for optimizer state.
        - If you later add a scheduler, save scheduler.state_dict() too.
    """
    pass


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """
    Input:
        model:
            Model object to load into.

        optimizer:
            Optimizer object to restore.

        checkpoint_dir:
            Directory created by save_checkpoint.

        device:
            Device used for torch.load map_location.

    Output:
        A tuple:
            model, optimizer, epoch, global_step

    What to do:
        1. Load model weights. There are two common choices:
            a. Recreate the model with AutoModelForCausalLM.from_pretrained.
            b. Or, if you already have the same architecture object, load a
               state_dict.
        2. Load training_state.pt with torch.load(..., map_location=device).
        3. Call optimizer.load_state_dict(...).
        4. Return restored epoch/global_step so training can resume.

    Things to watch:
        - HuggingFace's from_pretrained is usually the simplest model-loading
          path. This helper may become easier once you decide exactly how your
          training script constructs models.
        - Optimizer state can be large.
    """
    pass


def maybe_save_checkpoint(model, optimizer, config, epoch, global_step):
    """
    Input:
        model, optimizer:
            Current training objects.

        config:
            TrainConfig with checkpoint_every_steps and output_dir.

        epoch:
            Current epoch index.

        global_step:
            Number of optimizer updates completed so far.

    Output:
        Nothing.

    What to do:
        1. If config.checkpoint_every_steps is None, return immediately.
        2. If global_step is 0, return immediately.
        3. If global_step is divisible by checkpoint_every_steps:
            a. Build a checkpoint directory like:
                   Path(config.output_dir) / f"step-{global_step:06d}"
            b. Call save_checkpoint(...).
    """
    pass


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    grad_accum_steps=1,
    epoch=0,
    start_global_step=0,
    log_every_steps=10,
    use_wandb=False,
):
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

        epoch:
            Current epoch index, used only for logging.

        start_global_step:
            Number of optimizer updates already completed before this epoch.

        log_every_steps:
            Log every N optimizer updates.

        use_wandb:
            Whether to call wandb.log through log_train_metrics.

    Output:
        A tuple:
            avg_loss:
                Average unscaled loss across mini-batches for the epoch.

            global_step:
                Updated optimizer-step count after this epoch.

    What to do:
        1. Put the model in train mode with model.train().
        2. Clear old gradients with optimizer.zero_grad().
        3. For each micro-batch:
            a. Move it to device.
            b. Call outputs = model(**batch).
            c. Read the scalar loss from outputs.loss.
            d. Save loss.item() for logging before scaling it.
            e. Divide loss by grad_accum_steps.
            f. Call backward() on the scaled loss.
            g. Every grad_accum_steps micro-batches:
                - call optimizer.step()
                - increment global_step
                - optionally log metrics
                - call optimizer.zero_grad()
        4. If the epoch ends with leftover accumulated gradients, do one final
           optimizer.step().
        5. Return average unscaled loss and updated global_step.

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
    global_step = start_global_step
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        loss_value = loss.item()
        losses.append(loss_value)
        accum_steps += 1
        (loss / grad_accum_steps).backward()

        if accum_steps == grad_accum_steps:
            optimizer.step()
            global_step += 1

            if global_step % log_every_steps == 0:
                log_train_metrics(
                    {
                        "train/loss": loss_value,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    global_step=global_step,
                    use_wandb=use_wandb,
                )

            accum_steps = 0
            optimizer.zero_grad()

    if accum_steps > 0:
        optimizer.step()
        global_step += 1
        optimizer.zero_grad()

    if len(losses) == 0:
        return None, global_step

    return torch.mean(torch.tensor(losses)).item(), global_step


def train(model, dataloader, optimizer, device, config):
    """
    Input:
        model:
            HuggingFace causal LM.

        dataloader:
            PyTorch DataLoader yielding tokenized SFT batches.

        optimizer:
            torch optimizer.

        device:
            Usually "cuda" or "cpu".

        config:
            TrainConfig.

    Output:
        The final global_step.

    What to do:
        1. Initialize global_step = 0.
        2. Loop over range(config.num_epochs).
        3. Call train_one_epoch with config.grad_accum_steps, logging settings,
           and current global_step.
        4. Log epoch-level average loss.
        5. Save checkpoint at the end of each epoch, or call
           maybe_save_checkpoint if you want step-based checkpointing.
        6. Return global_step.

    Things to watch:
        - Keep step-based checkpointing and epoch-end checkpointing simple.
        - For the first overfit test, checkpointing can be disabled.
        - Once this works, you can add scheduler support in the same pattern.
    """
    pass


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
    loss, global_step = train_one_epoch(model, dataloader, optimizer, 'cuda', 1)
    print(f'Loss: {loss}, global_step: {global_step}')
    #print(f'Final tokens:\n{final_tokens}')
if __name__  == '__main__':
    run_tiny_train_smoke_test()
