import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# SIMPLE TOKENIZER (TEMP)
# -------------------------
class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.stoi = {}

    def build(self, text):
        chars = list(set(text))
        self.stoi = {ch: i % self.vocab_size for i, ch in enumerate(chars)}

    def encode(self, text):
        if not self.stoi:
            self.build(text)
        return [self.stoi.get(ch, 0) for ch in text]


# -------------------------
# DATASET
# -------------------------
class SpiderWebDataset(Dataset):
    def __init__(self, data_dir, tokenizer, seq_len=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        all_tokens = []
        files = glob.glob(os.path.join(data_dir, "*.txt"))

        if not files:
            raise FileNotFoundError(f"No .txt files found in {data_dir}")

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        self.num_blocks = len(self.tokens) // (seq_len + 1)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start_idx = idx * (self.seq_len + 1)
        chunk = self.tokens[start_idx: start_idx + self.seq_len + 1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


# -------------------------
# MAIN LOADER (FIXED)
# -------------------------
def get_dataloader(config, tokenizer=None, split="train"):
    data_path = os.path.join("data", "raw")

    # 🔥 fallback tokenizer
    if tokenizer is None:
        tokenizer = SimpleTokenizer(config.model.vocab_size)

    dataset = SpiderWebDataset(
        data_dir=data_path,
        tokenizer=tokenizer,
        seq_len=config.model.max_seq_len
    )

    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    return loader