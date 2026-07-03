"""
Data pipeline for the AttnRes LM.

Uses tiktoken (BPE tokenization, same as GPT-2/GPT-4) and HuggingFace datasets
for efficient data loading. Supports both streaming and non-streaming datasets.

For the prototype, we provide a simple text dataset that:
  - Tokenizes raw text using tiktoken BPE
  - Creates fixed-length sequences with packing (no waste)
  - Returns input_ids and labels (shifted by 1 for next-token prediction)
"""

from __future__ import annotations

import os
from typing import Optional, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken


# Default tokenizer: GPT-2 BPE (50257 tokens)
# For larger models, we'd use a custom 128K vocab tokenizer
DEFAULT_ENCODING = "gpt2"


def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Get a tiktoken BPE tokenizer."""
    return tiktoken.get_encoding(encoding_name)


class TextDataset(Dataset):
    """
    Tokenized text dataset with sequence packing.

    Takes raw text, tokenizes it, packs into fixed-length sequences,
    and returns (input_ids, labels) pairs for next-token prediction training.

    Args:
        texts: List of raw text strings, or a path to a text file.
        tokenizer: tiktoken Encoding (or name of encoding to load).
        max_length: Sequence length for packed sequences.
        cache_file: Optional path to cache tokenized data on disk.
    """

    def __init__(
        self,
        texts=None,
        tokenizer=None,
        max_length: int = 1024,
        cache_file: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        self.max_length = max_length

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = get_tokenizer()
        elif isinstance(tokenizer, str):
            self.tokenizer = get_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Load from cache or tokenize fresh
        if cache_file and os.path.exists(cache_file):
            self.tokens = torch.load(cache_file, map_location="cpu")
        else:
            if file_path is not None:
                with open(file_path, "r", encoding="utf-8") as f:
                    texts = f.read().split("\n\n")  # split on blank lines
            elif texts is None:
                raise ValueError("Either texts or file_path must be provided")

            self.tokens = self._tokenize_and_pack(texts)

            if cache_file:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                torch.save(self.tokens, cache_file)

    def _tokenize_and_pack(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenize all texts and pack into fixed-length sequences.

        Packing concatenates all tokenized text and splits into max_length chunks,
        eliminating padding waste. Each sequence is a contiguous slice of the corpus.
        """
        all_tokens = []
        for text in texts:
            # Encode with end-of-text token as document separator
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eot_token]
            all_tokens.extend(tokens)

        # Truncate to a multiple of max_length
        total_len = len(all_tokens)
        num_sequences = total_len // self.max_length
        all_tokens = all_tokens[: num_sequences * self.max_length]

        # Reshape to [num_sequences, max_length]
        tokens_tensor = torch.tensor(all_tokens, dtype=torch.long).view(-1, self.max_length)
        return tokens_tensor

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with input_ids and labels.

        Labels are the same as input_ids (the model internally shifts by 1
        for next-token prediction in the loss computation).
        """
        input_ids = self.tokens[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class StreamingTextDataset(Dataset):
    """
    Streaming dataset for large-scale training.

    Tokenizes and packs sequences on-the-fly from a HuggingFace dataset,
    allowing training on corpora too large to fit in memory.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "openwebtext", "wikitext")
        split: Dataset split ("train", "validation", etc.)
        tokenizer: tiktoken Encoding
        max_length: Sequence length
        max_sequences: Maximum number of sequences to produce (for limited training)
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "train",
        tokenizer=None,
        max_length: int = 1024,
        max_sequences: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.max_length = max_length

        if tokenizer is None:
            self.tokenizer = get_tokenizer()
        elif isinstance(tokenizer, str):
            self.tokenizer = get_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer

        from datasets import load_dataset

        ds = load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)

        # Get the text column
        text_col = "text" if "text" in ds.column_names else ds.column_names[0]

        # Tokenize and pack
        all_tokens = []
        for example in ds:
            text = example[text_col]
            if text.strip():  # skip empty lines
                tokens = self.tokenizer.encode(text) + [self.tokenizer.eot_token]
                all_tokens.extend(tokens)

            if max_sequences and len(all_tokens) >= max_sequences * self.max_length:
                break

        # Truncate and reshape
        num_sequences = len(all_tokens) // self.max_length
        all_tokens = all_tokens[: num_sequences * self.max_length]
        self.tokens = torch.tensor(all_tokens, dtype=torch.long).view(-1, self.max_length)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.tokens[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for LLM training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # important for consistent sequence packing
    )
