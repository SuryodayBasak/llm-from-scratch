import torch

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# Sample input tensor: this corresponds to input embeddings.
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]]
)

# Select the second vector of the input as the query.
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

# Compute attention scores.
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

"""
A few different ways to normalize the attention scores into attention weights.
These are for illustration and practice.
"""
print("\n--- Different ways to normalize attention scores ---")
# Normalize the attention scores -- simple way.
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights (simple sum): ", attn_weights_2_tmp)
print("Sum: ", attn_weights_2_tmp.sum())

# Attention weights using naive softmax.
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights (naive softmax):", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# Attention weights using PyTorch softmax.
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights (PyTorch softmax):", attn_weights_2)
print("Sum:", attn_weights_2.sum())

print("\n--- Compute the context vector corresponding to input[i] ---")
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("Context vector:", context_vec_2)

print("\n--- Computing attention weights for all input tokens ---")
print("Using loops:")

attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

print("\nUsing matrix multiplication:")
attn_scores = inputs @ inputs.T
print(attn_scores)

print("\nNormalizing attention scores using PyTorch softmax:")
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

print("\n--- Verifying that each row sums to 1 ---")
print(attn_weights.sum(dim=-1))  # Each row should sum to 1

print("\n--- Computing context vectors for all input token embeddings ---")
all_context_vectors = attn_weights @ inputs
print(all_context_vectors)