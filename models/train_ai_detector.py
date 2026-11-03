"""Fine-tune the AI-text detector named in the manuscript.

Checkpoint: microsoft/deberta-v3-large.
Package pin recorded in the manuscript: transformers 4.38.1.
Positive class: probability at least 0.72, the value selected by Youden's J
on the 3,000-row validation partition.

Human texts must fall in January 2019-October 2022. Synthetic prompts must
use question ids that are absent from the analytical sample. Optimization
settings are not printed and are read from hparams.json.
"""

from __future__ import annotations

import argparse

import pandas as pd

from manuscript_spec import DETECTOR_CHECKPOINT, DETECTOR_HUMAN_N, DETECTOR_SYNTHETIC_N
from models.hparams import require_hparams


def validate_training_frame(frame: pd.DataFrame, analytical_questions: set[str] | None) -> None:
    human = frame.loc[frame["label"] == 0]
    synthetic = frame.loc[frame["label"] == 1]
    if len(human) != DETECTOR_HUMAN_N or len(synthetic) != DETECTOR_SYNTHETIC_N:
        raise SystemExit(
            f"manuscript training counts are {DETECTOR_HUMAN_N} human and "
            f"{DETECTOR_SYNTHETIC_N} synthetic; file has {len(human)} and {len(synthetic)}"
        )
    dates = pd.to_datetime(human["answer_creation_date"], utc=True, errors="coerce")
    if dates.isna().any() or dates.min() < pd.Timestamp("2019-01-01", tz="UTC") or dates.max() >= pd.Timestamp("2022-11-01", tz="UTC"):
        raise SystemExit("human detector texts must be dated January 2019 through October 2022")
    if analytical_questions is not None:
        overlap = set(synthetic["prompt_question_id"].astype(str)).intersection(analytical_questions)
        if overlap:
            raise SystemExit("synthetic prompts overlap the analytical sample")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--analytical-ids")
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    hparams = require_hparams(args.hparams, "detector")
    if hparams["model_name"] != DETECTOR_CHECKPOINT:
        raise SystemExit(f"manuscript checkpoint is {DETECTOR_CHECKPOINT}")
    analytical = None
    if args.analytical_ids:
        analytical = set(pd.read_csv(args.analytical_ids)["question_id"].astype(str))
    train = pd.read_csv(args.train)
    validate_training_frame(train, analytical)
    validation = pd.read_csv(args.validation)
    if len(validation) != 3000:
        raise SystemExit("manuscript validation partition has 3,000 rows")
    try:
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("transformers is required to train the detector") from exc
    if transformers.__version__ != hparams["transformers_version"]:
        raise SystemExit(
            f"installed transformers is {transformers.__version__}; "
            f"manuscript reports {hparams['transformers_version']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_CHECKPOINT, num_labels=2)
    import torch

    class _Set(torch.utils.data.Dataset):
        def __init__(self, enc, y):
            self.enc = enc
            self.y = list(y)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            item = {key: torch.tensor(value[idx]) for key, value in self.enc.items()}
            item["labels"] = torch.tensor(int(self.y[idx]))
            return item

    encoded = tokenizer(train["text"].astype(str).tolist(), truncation=True, padding=True, max_length=512)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out_dir,
            learning_rate=float(hparams["learning_rate"]),
            num_train_epochs=float(hparams["num_epochs"]),
            per_device_train_batch_size=int(hparams["batch_size"]),
            seed=int(hparams["seed"]),
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=_Set(encoded, train["label"].to_numpy()),
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
