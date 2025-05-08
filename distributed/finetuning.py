

     
#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------
# Multi-node, multi-GPU RACE fine-tuning (memory-safe)
# -------------------------------------------------
from __future__ import annotations
import os, argparse, datasets, transformers, torch
from datasets import load_dataset, load_from_disk
from transformers import (AutoTokenizer, AutoModelForMultipleChoice,
                          get_linear_schedule_with_warmup, set_seed)
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from typing import Optional, Union, Dict, List
from accelerate import Accelerator
from tqdm.auto import tqdm


# ---------- distributed context ----------
acc = Accelerator()
world_size   = acc.num_processes
global_rank  = acc.process_index
local_rank   = acc.local_process_index
cache_dir    = "./cache_RACE"          # shared filesystem path

# ---------- tokeniser ----------
MODEL_NAME = "xlm-roberta-base"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
NO_ANSWER  = "None of the answers are correct."

def batched_tokenize(batch: Dict[str, List]) -> Dict[str, List]:
    """
    Batch-level preprocess.
    The output lists are stored on disk; no tensors are created here.
    """
    articles   = batch["article"]
    questions  = batch["question"]
    options    = batch["options"]
    answers    = batch["answer"]

    all_input_ids, all_masks, all_labels = [], [], []
    for art, q, opts, ans in zip(articles, questions, options, answers):
        # build question+option strings
        if "_" in q:                     # cloze
            qo = []
            for opt in opts:
                if opt == NO_ANSWER:
                    qo.append(opt)
                else:
                    filled = q
                    for w in opt.split(";"):
                        filled = filled.replace("_", f" {w} ", 1)
                    qo.append(" ".join(filled.split()))
        else:                            # normal question
            qo = [q + " " + opt for opt in opts]

        enc = tokenizer([art]*len(qo), qo,
                        padding="max_length",
                        truncation="only_first",
                        max_length=512)

        all_input_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
        all_labels.append(list("ABCD").index(ans))

    return {"input_ids": all_input_ids,
            "attention_mask": all_masks,
            "labels": all_labels}


def get_tokenised_dataset() -> datasets.DatasetDict:
    """
    Rank-0 downloads + tokenises once, saves Arrow cache.
    Other ranks wait, then memory-map the cache.
    """
    if acc.is_main_process:
        if not os.path.exists(cache_dir):
            print("[rank-0] downloading RACE …")
            raw = load_dataset("race", name="all")  # train + validation splits
            print("[rank-0] tokenising …")
            tokenised = raw.map(
                batched_tokenize,
                batched=True,
                remove_columns=raw["train"].column_names,
                num_proc=8,
                load_from_cache_file=True,
            )
            tokenised.save_to_disk(cache_dir)
            print("[rank-0] saved to", cache_dir)
    acc.wait_for_everyone()               # ------- barrier -------
    return load_from_disk(cache_dir)


@dataclass
class DataCollatorMC:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # each feature: {"input_ids": [num_opts, 512], ...}
        batch = {k: [f[k] for f in features] for k in features[0]}
        # stack option dimension, then batch dimension → (B, num_opts, 512)
        input_ids      = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])
        labels         = torch.tensor(batch["labels"], dtype=torch.long)
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}


def create_loaders(ds, args):
    collator = DataCollatorMC(tokenizer)
    train_loader = DataLoader(
        ds["train"],
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        ds["validation"],
        shuffle=False,
        batch_size=args.eval_batch_size,
        collate_fn=collator,
    )
    return train_loader, val_loader


def train(args):
    print(f"[init] world={world_size} rank={global_rank} local={local_rank} "
          f"host={os.uname().nodename}", flush=True)

    ds = get_tokenised_dataset()

    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

    train_loader, val_loader = create_loaders(ds, args)

    set_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ---------- barrier before DDP wrap ----------
    acc.wait_for_everyone()

    model, optimizer, train_loader, val_loader = acc.prepare(
        model, optimizer, train_loader, val_loader
    )

    num_updates = len(train_loader) * args.num_epochs
    warmup      = int(num_updates * 0.1)
    lr_sched    = get_linear_schedule_with_warmup(optimizer, warmup, num_updates)

    pbar = tqdm(range(num_updates), disable=not acc.is_main_process)
    best = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            out   = model(**batch)
            acc.backward(out.loss)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()
            pbar.update(1)

        # ---- evaluation ----
        model.eval()
        preds, refs = [], []
        for batch in val_loader:
            with torch.no_grad():
                logits = model(**batch).logits
            preds.append(acc.gather(logits.argmax(-1)))
            refs.append(acc.gather(batch["labels"]))
        preds = torch.cat(preds)[: len(ds["validation"])]
        refs  = torch.cat(refs) [: len(ds["validation"])]
        res   = metric.compute(predictions=preds, references=refs)

        acc.print(f"epoch {epoch}  val-acc={res['accuracy']:.4f}")
        if res["accuracy"] > best:
            best = res["accuracy"]
            acc.wait_for_everyone()
            unwrapped = acc.unwrap_model(model)
            if acc.is_main_process:
                unwrapped.save_pretrained("./best_model")
                tokenizer.save_pretrained("./best_model")
                acc.print(f"*** new best ({best:.4f}) – model saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--num_epochs",    type=int,   default=3)
    ap.add_argument("--train_batch_size", type=int, default=14)
    ap.add_argument("--eval_batch_size",  type=int, default=14)
    ap.add_argument("--seed",          type=int,   default=1234)
    args = ap.parse_args()
    train(args)
