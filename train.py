import os
import torch
from datasets import CharDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

DATA_DIR = "data"


if __name__ == "__main__":
    # parameters
    seq_length = 100
    batch_size = 64
    filename = "shakespeare"
    data_path = os.path.join(DATA_DIR, f"{filename}.txt")
    print(f"Data path: {data_path}")

    dataset = CharDataset(seq_length, data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
