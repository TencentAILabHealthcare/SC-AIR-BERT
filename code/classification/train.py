import sys
sys.path.append('./code/bert')

import torch
from torch.utils.data import DataLoader
from FCModel import FCModel
from Dataset import Dataset
from Dataset_singleSentence import Dataset_singleSentence


import model
from dataset import WordVocab

import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import os.path
import pandas as pd
import os
from os.path import isdir
from os import system

import nni
import logging
import random

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset")
parser.add_argument("-d", "--valid_dataset", required=True, type=str, help="valid dataset")
parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test dateset")
parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

parser.add_argument("-s", "--seq_len", type=int, default=80, help="maximum sequence len")
parser.add_argument("--prob", type=float, default=0.0, help="prob")

parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="min epochs")
parser.add_argument("--lr_b", type=float, default=1e-4, help="learning rate of adam")
parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate of adam")

parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
parser.add_argument("--with_cuda", type=bool,  default=True, help="")

parser.add_argument("--class_name", type=str, default=None, help="class name")
parser.add_argument("--bert_model", type=str, default=None, help="bert model")
parser.add_argument("--finetune", type=int, default=1, help="finetune bert")

parser.add_argument("--NNI_Search", type=int, default=1, help="NNI Search")
parser.add_argument("--in_features", type=int, default=256, help="in_features")
parser.add_argument("--chain", type=int, default=2, help="the number of chain")

parser.add_argument("--seed", type=int, default=6, help="default seed")
args = parser.parse_args()

class_name = args.class_name

if args.NNI_Search:
    print('Use NNI Search!')
    RCV_CONFIG = nni.get_next_parameter()

    seed = RCV_CONFIG['seed']
    path = os.path.join(args.output_path,class_name,'seed_'+seed)
else:
    path = os.path.join(args.output_path,class_name)
    seed = args.seed

train_dataset = args.train_dataset
valid_dataset = args.valid_dataset
test_dataset = args.test_dataset

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

setup_seed(seed)


print("Loading Vocab")
vocab = WordVocab.load_vocab(args.vocab_path)
print("Vocab Size: ", len(vocab))

if(args.chain==1):
    Dataset = Dataset_singleSentence

print("Loading Train Dataset")
train_dataset = Dataset(train_dataset, 
                            vocab, 
                            seq_len=args.seq_len,
                            on_memory=True,
                            prob = args.prob,
                            class_name = class_name)
valid_dataset = Dataset(valid_dataset, 
                            vocab, 
                            seq_len=args.seq_len,
                            on_memory=True,
                            prob = args.prob,
                            class_name = class_name)
test_dataset = Dataset(test_dataset, 
                            vocab, 
                            seq_len=args.seq_len,
                            on_memory=True,
                            prob = args.prob,
                            class_name = class_name)
print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32,shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=32)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=32)
print("data loaded")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("config finished")


bert_model = torch.load(args.bert_model)
bert_model = bert_model.to(device)


model = FCModel(in_features=args.in_features)
model = model.to(device)

if args.with_cuda and torch.cuda.device_count() > 1:
    print("Using %d GPUS" % torch.cuda.device_count())
    bert_model = torch.nn.DataParallel(bert_model, device_ids=args.cuda_devices)
    model = torch.nn.DataParallel(model, device_ids=args.cuda_devices)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_c)
bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr_b)

crit = torch.nn.BCELoss()


def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict)
    # print('predict:',predict)
    # print('label:',label)
    correct = (rounded_predict == label).float()
    accuracy = correct.sum() / len(correct)

    return accuracy


def train(dataset_loader,train):

    epoch_loss, epoch_acc = 0., 0.
    total_len = 0

    total_real = torch.empty(0,dtype=int)
    total_pred = torch.empty(0)
    total_ID = torch.empty(0,dtype=int)
    total_pred.to(device)
    total_real.to(device)

    for i, data in enumerate(dataset_loader):
        if(train):
            if(args.finetune):
                bert_model.train()
            else:
                bert_model.eval()
            model.train()
        else:
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
        loss = crit(predict, label.float())
        acc = binary_accuracy(predict, label)

        #gd
        if(train):
            optimizer.zero_grad() 
            if(args.finetune):
                bert_optimizer.zero_grad()
            loss.backward() 
            
            optimizer.step() 
            if(args.finetune):
                bert_optimizer.step()

        epoch_loss += loss * len(label)
        epoch_acc += acc * len(label)
        total_len += len(label)
        
        total_real = torch.cat([total_real.to(device),label],dim=0)
        total_pred = torch.cat([total_pred.to(device),predict],dim=0)
        total_ID = torch.cat([total_ID,ID],dim=0)

        # print("batch %d loss:%f accuracy:%f" % (i, loss, acc))

    auc = roc_auc_score(total_real.detach().cpu().numpy(),total_pred.detach().cpu().numpy())

    return epoch_loss/total_len, epoch_acc/total_len, auc, total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy(), total_ID.numpy()

# early stop
def stop_check(loss,stop_criterion,stop_criterion_window):
    w = loss[-stop_criterion_window:]
    # print(w)
    if((w[0]-w[-1])/w[0] < stop_criterion):
        return 1
    else:
        return 0

# plot loss&auc
def plot(Loss_list,Accuracy_list,outputname,name,x):

    x1 = range(0, len(Loss_list))
    x2 = range(0, len(Accuracy_list))
    y1 = Loss_list
    y2 = Accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)

    plt.title(name)
    plt.ylabel('loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)

    plt.xlabel(x)
    plt.ylabel('auc')
    plt.savefig(outputname)
    plt.close()


print("Training Start")
index = 0

train_loss_total,val_loss_total = [],[]
stop_check_list = []
epoch = 0
epochs_min = 10

min_loss = 100
max_auc = 0
min_loss_auc,min_loss_acc = 0,0
max_auc_auc,max_auc_acc = 0,0
last_epoch_auc,last_epoch_acc = 0,0

epoch_train_loss_list = []
epoch_train_auc_list = []
epoch_valid_loss_list = []
epoch_valid_auc_list = []
epoch_test_loss_list = []
epoch_test_auc_list = []

if not(isdir(path)):
    cmd = 'mkdir -p ' + path
    system(cmd)

while(True):
    epoch_train_loss, epoch_train_acc,epoch_train_auc, epoch_train_real, epoch_train_pred, epoch_train_ID = train(train_data_loader,train=True)
    epoch_train_loss_list.append(epoch_train_loss)
    epoch_train_auc_list.append(epoch_train_auc)
    index += 1
    print("EPOCH %d_train loss:%f accuracy:%f auc:%f" % (index, epoch_train_loss, epoch_train_acc, epoch_train_auc))
    
    train_loss_total.append(epoch_train_loss)
    stop_criterion = 0.001
    stop_criterion_window = 10

    with torch.no_grad():
        epoch_valid_loss, epoch_valid_acc,epoch_valid_auc,epoch_valid_real, epoch_valid_pred, epoch_valid_ID = train(valid_data_loader,train=False)
        epoch_valid_loss_list.append(epoch_valid_loss)
        epoch_valid_auc_list.append(epoch_valid_auc)
        print("EPOCH %d_valid loss:%f accuracy:%f auc:%f" % (index, epoch_valid_loss, epoch_valid_acc, epoch_valid_auc))
        
        val_loss_total.append(epoch_valid_loss)

        epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_real, epoch_test_pred, epoch_test_ID = train(test_data_loader,train=False)
        epoch_test_loss_list.append(epoch_test_loss)
        epoch_test_auc_list.append(epoch_test_auc)
        print("EPOCH %d_test loss:%f accuracy:%f auc:%f" % (index, epoch_test_loss, epoch_test_acc,epoch_test_auc))

        if(min_loss > epoch_valid_loss):
            min_loss = epoch_valid_loss
            min_loss_auc = epoch_test_auc
            min_loss_acc = epoch_test_acc

            model_output = os.path.join(path,'min_loss_model.pth')
            state = {
                'bert_model':bert_model.state_dict(),
                'fc_model':model.state_dict()
            }
            torch.save(state, model_output)
            # model.to(device)

            data = {
                'ID':epoch_test_ID,
                'real':epoch_test_real.tolist(),
                'pred':epoch_test_pred.tolist()
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(path,'min_loss_result.csv'),index=None)

        if(max_auc<epoch_valid_auc):
            max_auc = epoch_valid_auc
            max_auc_auc = epoch_test_auc
            max_auc_acc = epoch_test_acc
            model_output = os.path.join(path,'max_auc_model.pth')
            state = {
                'bert_model':bert_model.state_dict(),
                'fc_model':model.state_dict()
            }
            torch.save(state, model_output)
            # model.to(device)

            data = {
                'ID':epoch_test_ID,
                'real':epoch_test_real.tolist(),
                'pred':epoch_test_pred.tolist()
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(path,'max_auc_result.csv'),index=None)
            

        if epoch > epochs_min:
            if val_loss_total:
                stop_check_list.append(stop_check(val_loss_total, stop_criterion, stop_criterion_window))
                if np.sum(stop_check_list[-3:]) >= 3:

                    last_epoch_auc = epoch_test_auc
                    last_epoch_acc = epoch_test_acc
                    model_output = os.path.join(path,'last_epoch_model.pth')
                    state = {
                        'bert_model':bert_model.state_dict(),
                        'fc_model':model.state_dict()
                    }
                    torch.save(state, model_output)
                    # model.to(device)

                    data = {
                        'ID':epoch_test_ID,
                        'real':epoch_test_real.tolist(),
                        'pred':epoch_test_pred.tolist()
                    }
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(path,'last_epoch_result.csv'),index=None)
                    break
            
        # if(epoch==5):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_c*0.1)
        #     bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr_b*0.1)


    # if(epoch>29):
    #     break
    epoch += 1

auc_csv = pd.DataFrame(columns=['lr_b','lr_c'])
auc_csv['max_auc_auc'] = [max_auc_auc]
auc_csv['max_auc_acc'] = [max_auc_acc.item()]

auc_csv['min_loss_auc'] = [min_loss_auc]
auc_csv['min_loss_acc'] = [min_loss_acc.item()]

auc_csv['last_epoch_auc'] = [last_epoch_auc]
auc_csv['last_epoch_acc'] = [last_epoch_acc.item()]

auc_csv['bert_model'] = args.bert_model
auc_csv['finetune'] = args.finetune
auc_csv['prob'] = args.prob
auc_csv['seq_len'] = args.seq_len
auc_csv['batch_size'] = args.batch_size
auc_csv['lr_b'] = args.lr_b
auc_csv['lr_c'] = args.lr_c

auc_csv.to_csv(os.path.join(path,'parameters.csv'),index=None)

nni.report_final_result(min_loss_auc)

plot(epoch_train_loss_list,epoch_train_auc_list,os.path.join(path,'train_loss_auc.png'),class_name + '_train','epochs')
plot(epoch_valid_loss_list,epoch_valid_auc_list,os.path.join(path,'valid_loss_auc.png'),class_name + '_valid','epochs')
plot(epoch_test_loss_list,epoch_test_auc_list,os.path.join(path,'test_loss_auc.png'),class_name + '_test','epochs')

print('auc:',min_loss_auc)
print('acc:',min_loss_acc.item())



