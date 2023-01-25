import torch
from torch.utils.data import DataLoader
from FCModel import FCModel
from Dataset import Dataset
from Dataset_singleSentence import Dataset_singleSentence

import sys
sys.path.append('./code/bert')
import model
from dataset import WordVocab

from model import BERT

import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import os.path
import pandas as pd
import os
from os.path import isdir
from os import system

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test dateset")

parser.add_argument("-s", "--seq_len", type=int, default=80, help="maximum sequence len")
parser.add_argument("--class_name", type=str, default=None, help="class name")
parser.add_argument("--load_model", type=str, default=None, help="load model")
parser.add_argument("-o", "--output_path", required=True, type=str, help="output")

args = parser.parse_args()

vocab = WordVocab.load_vocab(args.vocab_path)

test_dataset = Dataset(args.test_dataset, 
                            vocab, 
                            seq_len=args.seq_len,
                            on_memory=True,
                            prob = 0.0,
                            class_name = args.class_name)
test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model = BERT(len(vocab), hidden=512, n_layers=6, attn_heads = 4,max_len=args.seq_len,embedding_mode='normal')

model = FCModel(in_features=512)


checkpoint = torch.load(args.load_model)
model.load_state_dict(checkpoint['fc_model'])
bert_model.load_state_dict(checkpoint['bert_model'])
model = model.to(device)
bert_model.to(device)

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
        bert_model.eval()
        model.eval()
        
        label = data['classification_label']
        label = label.cuda()

        encoding = data['bert_input']
        segment_info = data['segment_label']
        ID = data['ID']
        bert_output = bert_model(encoding.to(device),segment_info.to(device))
 
        pooler_output = bert_output[:,0,:]
        
        predict = model(pooler_output).squeeze()
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