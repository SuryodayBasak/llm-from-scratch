import re

class SimpleVocab:
    def __init__(self, text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        all_tokens = sorted(set(preprocessed))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])

        self.vocab = {token:integer for integer, token in enumerate(all_tokens)}
    
    def getVocab(self):
        return self.vocab

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v:k for k,v in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[id] for id in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

def main():
    text_path = 'data/the-verdict.txt'
    vocab = SimpleVocab(text_path)
    
    tokenizer = SimpleTokenizerV1(vocab.getVocab())
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."

    # Convert text to token IDs.
    ids = tokenizer.encode(text)
    print("Encoded IDs:", ids)

    # Convert token IDs back to text.
    decoded = tokenizer.decode(ids)
    print("Decoded text:", decoded)

if __name__ == "__main__":
    main()