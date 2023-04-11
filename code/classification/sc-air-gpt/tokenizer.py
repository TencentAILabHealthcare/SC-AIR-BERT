from tokenizers import Tokenizer, models, pre_tokenizers, trainers, ByteLevelBPETokenizer
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=['./data/pretrain/3-mer_sep/full_data_sep_pad.txt'], vocab_size=20000, min_frequency=1, special_tokens=["[SEP]","[PAD]"])

# Save files to disk
tokenizer.save_model("./data/pretrain/3-mer_sep_pad", "gpt_tokenizer")


