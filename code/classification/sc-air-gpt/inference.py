import torch
from torch.utils.data import DataLoader
from FCModel import FCModel
from Dataset import Dataset

import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import os.path
import pandas as pd
import os
from os.path import isdir
from os import system


import matplotlib.pyplot as plt

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test dateset")
parser.add_argument("--test_label", type=str, default=None, help="test dateset")
parser.add_argument("--block_size", type=int, default=80, help="maximum sequence len")
parser.add_argument("--vocab_file", type=str, default=None, help="gpt model")
parser.add_argument("--merges_file", type=str, default=None, help="gpt model")
parser.add_argument("--class_name", type=str, default=None, help="class name")
parser.add_argument("--load_model", type=str, default=None, help="load model")
parser.add_argument("-o", "--output_path", required=True, type=str, help="output")


args = parser.parse_args()

# tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", vocab_file=args.vocab_file,merges_file=args.merges_file,pad_token="[PAD]")
tokenizer = GPT2Tokenizer.from_pretrained("./code/classification/sc-air-gpt/gpt2_tokenizer", vocab_file=args.vocab_file,merges_file=args.merges_file,pad_token="[PAD]")
tokenizer.add_special_tokens({"additional_special_tokens": ["[SEP]"]})

test_dataset = Dataset(tokenizer, 
                            args.test_dataset, 
                            args.block_size,
                            args.test_label,
                            args.class_name)

test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
gpt_model = GPT2LMHeadModel(config=model_config)

model = FCModel(in_features=512,block_size=77)

checkpoint = torch.load(args.load_model)
model.load_state_dict(checkpoint['fc_model'])
gpt_model.load_state_dict(checkpoint['gpt_model'])
model = model.to(device)
gpt_model.to(device)

def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict) 
    correct = (rounded_predict == label).float()
    accuracy = correct.sum() / len(correct)

    return accuracy

def train(dataset_loader,train):

    epoch_acc = 0.
    total_len = 0

    total_real = torch.empty(0,dtype=int)
    total_pred = torch.empty(0)
    total_ID = torch.empty(0,dtype=int)
    total_pred.to(device)
    total_real.to(device)

    for i, data in enumerate(dataset_loader):
        gpt_model.eval()
        model.eval()
        
        # print(data['ID'])
        ID = data['ID']
        # print(ID)
        # print(ID)
        label = data['labels']
        # label = torch.tensor(label)
        label = label.cuda()
        # print(label)

        encoding = data['text']['input_ids']
        gpt_output = gpt_model(encoding.to(device),output_hidden_states=True).hidden_states[-1]
        # print(gpt_output)
        # print(gpt_output.size()) 
        # [batch_size,77,512]
        
        # pooler_output = bert_output[:,0,:]
        
        predict = model(gpt_output).squeeze()
        acc = binary_accuracy(predict, label)

        epoch_acc += acc * len(label)
        total_len += len(label)
        
        total_real = torch.cat([total_real.to(device),label],dim=0)
        total_pred = torch.cat([total_pred.to(device),predict],dim=0)
        total_ID = torch.cat([total_ID,ID],dim=0)


    auc = roc_auc_score(total_real.detach().cpu().numpy(),total_pred.detach().cpu().numpy())
    # auc = 0

    return epoch_acc/total_len, auc, total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy(), total_ID.numpy()

epoch_test_auc_list = []

epoch_test_acc,epoch_test_auc,epoch_test_real, epoch_test_pred, epoch_test_ID = train(test_data_loader,train=False)
epoch_test_auc_list.append(epoch_test_auc)
print("accuracy:%f auc:%f" % (epoch_test_acc,epoch_test_auc))

data = {
    'ID':epoch_test_ID,
    'real':epoch_test_real.tolist(),
    'pred':epoch_test_pred.tolist()
}
df = pd.DataFrame(data)
os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok = True) 
df.to_csv(args.output_path, index=None)