# https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
# https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(SelfAttention, self).__init__()
        ...

class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(CrossAttention, self).__init__()
        ...