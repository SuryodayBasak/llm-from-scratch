import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Akwirw ier"

encoded_tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Encoded tokens:", encoded_tokens)

# Decoding individual tokens:
for token in encoded_tokens:
    print(f"Token ID: {token}, Decoded: '{tokenizer.decode([token])}'")

decoded_text = tokenizer.decode(encoded_tokens)
print("Decoded text:", decoded_text)