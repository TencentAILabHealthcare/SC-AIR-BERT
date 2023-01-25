import argparse

from torch.utils.data import DataLoader

from model import BERT
from trainer import BERTTrainer_MLM
from dataset import WordVocab
from dataset import BERTDataset_MLM, BERTdataset_MLM_maskNeighbours
from dataset import BERTdataset_MLM_singleSentence, BERTdataset_MLM_singleSentence_maskNeighbours

import random
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

import nni
import logging
from os import system
from os.path import isdir

def plot(Loss_list,Accuracy_list,outputname,name,x):

    x1 = range(0, len(Loss_list))
    x2 = range(0, len(Accuracy_list))
    y1 = Loss_list
    y2 = Accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)
    #     plt.plot(x1, y1, 'o-')
    plt.title(name)
    plt.ylabel('loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)
    #  plt.plot(x2, y2, '.-')
    plt.xlabel(x)
    plt.ylabel('acc')
    plt.savefig(outputname)
    plt.close()

def set_logger(OutputFilePath):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(OutputFilePath)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def save_checkpoint(state, is_best, args, epoch, ACC_S, ACC_T):
    filename = 'checkpoint_epoch-{}_ACC-S-{:.3f}-ACC-T-{:.3f}.pth.tar'.format(epoch, ACC_S, ACC_T)
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))
    return dir_save_file

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
    # print('-'*30)
    # print(f'Seed is :{seed}')
    # print('-'*30)

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")
    # parser.add_argument("-output", "--output", required=True, type=str, help="output_path")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--seed", type=int, default=5, help="default seed")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--NNI_Search", type=int, default=1, help="use NNI search")
    parser.add_argument("--prob", type=float, default=0.10, help="probability of mask a word")
    parser.add_argument("--embedding_mode", type=str, default='normal', help="embedding mode")
    parser.add_argument("--process_mode", type=str, default='MLM', help="process dataset")
    args = parser.parse_args()

    # ImportantLogger = set_logger()

    if args.NNI_Search:
        print('Use NNI Search!')
        RCV_CONFIG = nni.get_next_parameter()
        # ImportantLogger.debug(RCV_CONFIG)
        
        lr = RCV_CONFIG['lr']
        epochs = RCV_CONFIG['epochs']
        hidden = RCV_CONFIG['hidden']
        layers = RCV_CONFIG['layers']
        batch_size = RCV_CONFIG['batch_size']
        prob = RCV_CONFIG['prob']
        attn_heads = RCV_CONFIG['attn_heads']
    else:
        lr = args.lr
        epochs = args.epochs
        hidden = args.hidden
        layers = args.layers
        batch_size = args.batch_size
        prob = args.prob
        attn_heads = args.attn_heads

    #-----------------
    setup_seed(args.seed)
    #-----------------
    print('batch_size:',batch_size)
    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))
    
    if(args.process_mode == 'MLM'):
        BERTDataset = BERTDataset_MLM
    elif(args.process_mode == 'MLM_MN'):
        BERTDataset = BERTdataset_MLM_maskNeighbours
    elif(args.process_mode == 'MLM_SS'):
        BERTDataset = BERTdataset_MLM_singleSentence
    elif(args.process_mode == 'MLM_SS_MN'):
        BERTDataset = BERTdataset_MLM_singleSentence_maskNeighbours
    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, 
                                vocab, 
                                seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, 
                                on_memory=args.on_memory,
                                prob = prob)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, 
                               vocab, 
                               seq_len=args.seq_len, 
                               on_memory=args.on_memory,
                               prob = prob) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads = attn_heads,max_len=args.seq_len,embedding_mode=args.embedding_mode)
    # bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=args.attn_heads,max_len=46)

    print("Creating BERT Trainer")
    trainer = BERTTrainer_MLM(bert, 
                          len(vocab), 
                          train_dataloader=train_data_loader, 
                          test_dataloader=test_data_loader,
                          lr=lr, 
                          betas=(args.adam_beta1, args.adam_beta2), 
                          weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, 
                          cuda_devices=args.cuda_devices, 
                          log_freq=args.log_freq)

    print("Training Start")
    history = {'loss_train':[], 'acc_train':[], 'loss_test':[] ,'acc_test': [],
    'epoch_loss_train':[], 'epoch_loss_test':[], 'epoch_acc_train':[], 'epoch_acc_test':[]} 

    epoch_acc_train_top = 0
    # path = os.path.join(args.output_path,'epochs' + str(epochs) + '_lr'+str(lr) + '_hidden' + str(hidden)
    # + '_layers' + str(layers) + '_batchsize' + str(batch_size) + '_prob' +str(prob))
    path = os.path.join(args.output_path,'hidden' + str(hidden)
    + '_layers' + str(layers) + '_attn_heads'+ str(attn_heads) + '_batchsize' + str(batch_size) + '_prob' +str(prob) + '_lr' + str(lr))
    
    if not(isdir(path)):
        cmd = 'mkdir -p ' + path
        system(cmd)

    parameters = {
        'train_dataset':[args.train_dataset],
        'test_dataset':[args.test_dataset],
        'vocab_path':[args.vocab_path],
        'hidden':[hidden],
        'layers':[layers],
        'attn_heads':[attn_heads],
        'seq_len':[args.seq_len],
        'batch_size':[batch_size],
        'epochs':[epochs],
        'seed':[args.seed],
        'num_workers':[args.num_workers],
        'lr':[lr],
        'prob':[prob],
        'embedding_mode':[args.embedding_mode],
    } 

    df_parameters = pd.DataFrame(parameters)
    df_parameters.to_csv(path + '/parameters.csv',index=None)
    
    for epoch in range(epochs):
        h = trainer.train(epoch)
        trainer.save(epoch, os.path.join(path,'bert.model'))

        history['loss_train'] += h['loss']
        history['acc_train'] += h['acc']

        history['epoch_loss_train'].append(h['epoch_loss'])
        history['epoch_acc_train'].append(h['epoch_acc'])
        if(epoch_acc_train_top<h['epoch_acc']):
            epoch_acc_train_top = h['epoch_acc']


        if test_data_loader is not None:
            h = trainer.test(epoch)

            history['loss_test'] += h['loss']
            history['acc_test'] += h['acc']

            history['epoch_loss_test'].append(h['epoch_loss'])
            history['epoch_acc_test'].append(h['epoch_acc'])
    
    nni.report_final_result(epoch_acc_train_top)
    
    # plot
    # path = args.output
    plot(history['loss_train'], history['acc_train'] ,path + '/loss_acc_train.png','loss_acc_train','iteration')
    plot(history['loss_test'], history['acc_test'], path + '/loss_acc_test.png','loss_acc_test','iteration')
    plot(history['epoch_loss_train'], history['epoch_acc_train'],path + '/loss_acc_epoch_train.png','loss_acc_epoch_train','epoch')
    plot(history['epoch_loss_test'], history['epoch_acc_test'], path + '/loss_acc_epoch_test.png','loss_acc_epoch_test','epoch')
    
    train_data = {
        'loss':history['loss_train'],
        'acc':history['acc_train']
    }
    test_data = {
        'loss':history['loss_test'],
        'acc':history['acc_test']
    }
    epoch_train_data = {
        'loss':history['epoch_loss_train'],
        'acc':history['epoch_acc_train']
    }
    epoch_test_data = {
        'loss':history['epoch_loss_test'],
        'acc':history['epoch_acc_test']
    }

    train_loss_acc = pd.DataFrame(train_data)
    train_loss_acc.to_csv(path + '/train_loss_acc.csv',index=None)
    test_loss_acc = pd.DataFrame(test_data)
    test_loss_acc.to_csv(path + '/test_loss_acc.csv',index=None)
    epoch_train_loss_acc = pd.DataFrame(epoch_train_data)
    epoch_train_loss_acc.to_csv(path + '/epoch_train_loss_acc.csv',index=None)
    epoch_test_loss_acc= pd.DataFrame(epoch_test_data)
    epoch_test_loss_acc.to_csv(path + '/epoch_test_loss_acc.csv',index=None)

if __name__ == '__main__':
    train()
