import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class CharDataset(Dataset):
    def __init__(self, data_path: str, seq_length: int = 100, device: str = "cpu"):
        self.seq_length = seq_length
        self.device = device

        # get the raw data and unique characters in it
        self.raw_data = self._read_data(data_path)
        self.vocab = self.get_vocab(self.raw_data)

        # build the vocab-to-int mappings
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        # encode the data and delete the raw string text
        self.encoded_data = self.encode(self.data) 
        del self.data

    def get_vocab(self, data: str) -> List[str]:
        """Return a list of unique characters in the data"""
        return list(set(data))

    def encode(self, data: str) -> List[int]:
        """Tokenize the data into a list of ints"""
        return [self.char2idx[char] for char in data]

    def decode(self, data: List[int]) -> str:
        """Decode a list of ints into a string"""
        return "".join([self.idx2char[idx] for idx in data])

    def __len__(self):
        return len(self.encoded_data) - self.seq_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single sample from the dataset"""
        inputs = self.encoded_data[idx:idx+self.seq_length] # context window
        targets = self.encoded_data[idx+self.seq_length]    # next character
        return torch.tensor(inputs, dtype=torch.long).to(self.device), torch.tensor(targets, dtype=torch.long).to(self.device)

    def _read_data(self, data_path: str) -> str:
        """Read the data from a file and return the raw data and vocab"""
        print(f"Reading data from {data_path}")
        try:
            with open(data_path, "r") as f:
                raw_data = f.read()
        except:
            raise FileNotFoundError(f"Could not find file at {data_path}")
        return raw_data

    def collate_fn(self, batch: List[str]):
        """Collate function to combine data samples into batches"""
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs)
        targets = torch.tensor(targets, dtype=torch.long).to(self.device)
        return inputs, targets


#     def prepare_data(self, seq_length):
#         n, p = 0, 0
#         # prepare inputs (we're sweeping from left to right in steps seq_length long)
#         if p+seq_length+1 >= len(data) or n == 0: 
#             hprev = np.zeros((hidden_size,1)) # reset RNN memory
#             p = 0 # go from start of data
#         inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
#         targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
           