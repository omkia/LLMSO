"""Train one ICI or ECI text classifier.

The manuscript defines the scores as locked out-of-sample probabilities
from two separate classifiers. Training uses answer text and the human
binary code only. Votes, acceptance, reputation, and post-event outcomes
are excluded. The encoder and optimization settings are not printed;
hparams.json must supply them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.hparams import assert_text_only, require_hparams


def binary_target(frame: pd.DataFrame, item_prefix: str, cut: float) -> pd.Series:
    cols = [col for col in frame.columns if col.startswith(item_prefix)]
    if not cols:
        raise SystemExit(f"no {item_prefix} item columns in the annotation file")
    return (frame[cols].mean(axis=1) >= cut).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--texts", required=True, help="CSV with item_id and answer_text.")
    parser.add_argument("--target", required=True, choices=("ici", "eci"))
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    hparams = require_hparams(args.hparams, "ici_eci")
    annotations = pd.read_csv(args.annotations)
    texts = pd.read_csv(args.texts)
    assert_text_only([col for col in texts.columns if col not in ("item_id", "answer_text", "split")])
    prefix = args.target + "_"
    item_cols = [col for col in annotations.columns if col.startswith(prefix)]
    items = annotations.groupby("item_id", as_index=False)[item_cols].mean()
    if "split" in annotations.columns:
        split = (
            annotations.groupby("item_id")["split"]
            .agg(lambda values: values.mode().iloc[0])
            .reset_index(name="split")
        )
        items = items.merge(split, on="item_id", how="left")
    items[args.target] = binary_target(items, prefix, float(hparams["positive_if_mean_ge"]))
    merged = texts.merge(items[["item_id", args.target] + (["split"] if "split" in items.columns else [])],
                         on="item_id", how="inner")
    if "split" not in merged.columns:
        raise SystemExit("the annotation file or the text file must carry the locked split")
    train = merged.loc[merged["split"] == "train"]
    if train.empty:
        raise SystemExit("the locked train split is empty")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("transformers is required to train the text classifiers") from exc
    tokenizer = AutoTokenizer.from_pretrained(hparams["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(hparams["model_name"], num_labels=2)
    encoded = tokenizer(train["answer_text"].astype(str).tolist(), truncation=True, padding=True, max_length=512)
    # The Trainer consumes a torch dataset. Import here so import of this
    # module does not require torch.
    import torch

    class _Set(torch.utils.data.Dataset):
        def __init__(self, enc, y):
            self.enc = enc
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.enc.items()}
            item["labels"] = torch.tensor(int(self.y[idx]))
            return item

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out),
            learning_rate=float(hparams["learning_rate"]),
            num_train_epochs=float(hparams["num_epochs"]),
            per_device_train_batch_size=int(hparams["batch_size"]),
            seed=int(hparams["seed"]),
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=_Set(encoded, train[args.target].to_numpy()),
    )
    trainer.train()
    trainer.save_model(out / args.target)
    tokenizer.save_pretrained(out / args.target)


if __name__ == "__main__":
    main()
