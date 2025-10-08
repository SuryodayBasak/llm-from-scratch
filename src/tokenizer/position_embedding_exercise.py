import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from data_loader_1 import create_dataloader_v1, GPTDatasetV1

def main():
    vocab_size = 50257
    output_dim = 256
    max_length = 4
    context_length = max_length

    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    dataloader = create_dataloader_v1(raw_text, batch_size=8,
                                    max_length=max_length, stride=max_length,
                                    shuffle=False)
    
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("\nToken Embeddings shape:\n", token_embeddings.shape)
    print("\nPosition Embeddings shape:\n", pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print("\nInput Embeddings shape:\n", input_embeddings.shape)

if __name__ == "__main__":
    main()