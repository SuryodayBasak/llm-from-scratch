import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        " of someunknownPlace."
        )

encoded_tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Encoded tokens:", encoded_tokens)

decoded_text = tokenizer.decode(encoded_tokens)
print("Decoded text:", decoded_text)