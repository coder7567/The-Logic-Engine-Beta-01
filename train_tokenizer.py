import os
import json
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers, Regex
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel


# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "corpus_raw.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "src", "python", "logic_engine", "tokenizer.json")

def data_iterator(filepath):
    """Generator to yield raw text from the JSONL corpus."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    payload = json.loads(line)
                    yield payload["text"]
                except json.JSONDecodeError:
                    continue

def build_logic_engine_tokenizer():
    print("Initializing Custom BPE Architecture...")
    # Initialize a Byte-Pair Encoding model
    tokenizer = Tokenizer(models.BPE())
    
    # Define hard boundaries (Pre-tokenization)
    # This prevents the BPE algorithm from destructively merging critical logic boundaries
    # with adjacent alphanumeric characters.
    
    # 1. Protect structural whitespace and newlines natively via ByteLevel
    # ByteLevel converts all bytes to printable characters, preserving spacing flawlessly.
    pre_toks = [ByteLevel(add_prefix_space=False)]
    
    # 2. Hard split on LaTeX blocks and mathematical symbols using Regex
    # We want these kept intact before BPE runs, forcing them to become standalone tokens or cleanly segmented.
    math_regex = Regex(r"(\\begin\{[^}]+\}|\\end\{[^}]+\}|\\[a-zA-Z]+|∑|∫|{|}|\[|\]|\(|\)|\||&|\+|-|\*|/|=|<|>)")
    pre_toks.append(Split(math_regex, behavior="isolated"))
    
    # 3. Hard split on Code Indentation (4 spaces, tabs, double newlines)
    indent_regex = Regex(r"( {4}|\t|\n\n|\n)")
    pre_toks.append(Split(indent_regex, behavior="isolated"))

    tokenizer.pre_tokenizer = Sequence(pre_toks)
    
    # Configure the Trainer
    # We heavily prioritize the explicit mathematical and programming tokens
    special_tokens = [
        "<s>", "<pad>", "</s>", "<unk>", "<mask>",
        "<|domain_math|>", "<|domain_cs|>", "<|domain_data|>", "<|persona_sys|>"
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=65536, # Standard highly optimized vocab size
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train the tokenizer
    print(f"Executing BPE training on corpus: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Corpus not found at {DATA_PATH}")
        
    tokenizer.train_from_iterator(data_iterator(DATA_PATH), trainer=trainer)
    
    # Configure the post-processor to handle ByteLevel decoding
    tokenizer.decoder = decoders.ByteLevel()
    
    # Save the architecture
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tokenizer.save(OUT_PATH)
    print(f"Tokenizer compilation complete. Serialized to: {OUT_PATH}")

if __name__ == "__main__":
    build_logic_engine_tokenizer()
