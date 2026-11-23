"""Fine-tune a SentenceTransformer on anchor-positive pairs with MNLR."""

import random
from pathlib import Path


def fine_tune(
    base_model: str,
    pairs: list[dict],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    """Train with MultipleNegativesRankingLoss and save the model to `output_dir`.

    Batches never hold two copies of the same positive (NO_DUPLICATES): MNLR uses the
    other positives in a batch as negatives, and both datasets reuse positives.
    """
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers import losses
    from sentence_transformers.training_args import BatchSamplers

    random.seed(seed)
    torch.manual_seed(seed)
    model = SentenceTransformer(base_model)
    train = Dataset.from_dict(
        {"anchor": [p["anchor"] for p in pairs], "positive": [p["positive"] for p in pairs]}
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train,
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    model.save(str(output_dir))

    return model
