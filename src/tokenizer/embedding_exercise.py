import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("Embedding weights:\n", embedding_layer.weight)

print("\n")
print(embedding_layer(input_ids))