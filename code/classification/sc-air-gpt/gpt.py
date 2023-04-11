from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vocab_file="./data/pretrain/3-mer_sep_pad/gpt_tokenizer-vocab.json"
merges_file="./data/pretrain/3-mer_sep_pad/gpt_tokenizer-merges.txt"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2_tokenizer", vocab_file=vocab_fi le,merges_file=merges_file,pad_token="[PAD]")
tokenizer.add_special_tokens({"additional_special_tokens": ["[SEP]"]})

print(tokenizer.encode("CAL ALK LKG KGP GPG PGT GTY TYK YKY KYI YIF [SEP] CAL ALK LKG KGP GPG PGT GTY TYK YKY KYI YIF [PAD]"))

train_data = './data/pretrain/3-mer_sep_pad/train_sep_pad.txt'
test_data = './data/pretrain/3-mer_sep_pad/test_sep_pad.txt'
# train_data = './data/pretrain/test_sep_pad/train.txt'
# test_data = './data/pretrain/test_sep_pad/test.txt'
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_data,
    block_size=77, 
)
eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=test_data,
    block_size=77, 
)


# load model configuration
model_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions= 77, 
    n_ctx=77, 
    n_embd=512, 
    n_layer=6,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    output_hidden_states=True,
)

# load model
model = GPT2LMHeadModel(config=model_config)

# set training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy='epoch', 
#     eval_steps=1,
#     save_steps=1,
#     num_train_epochs=50,
#     per_device_train_batch_size=128,
#     per_device_eval_batch_size=128,
#     learning_rate=1e-4,
#     weight_decay=0.01,
#     push_to_hub=False,
#     logging_dir='./logs',
# )

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=50,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=1e-4,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='./logs',
)

# load data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# set trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# train model
trainer.train()
