from torch.utils.data import Dataset
from itertools import islice

import tqdm
import torch
import random

import sys
sys.path.append('./code/bert')
from dataset import BERTDataset_MLM, WordVocab


class Dataset(BERTDataset_MLM):
    def __init__(self, corpus_path, vocab, seq_len, class_name, encoding="utf-8", corpus_lines=None, on_memory=True,prob=0.10):
        super().__init__(corpus_path, vocab, seq_len,encoding, corpus_lines, on_memory,prob)
        self.class_name = class_name

    def __getitem__(self, item):
        t0,t1, t2, t3 = self.get_corpus_line(item)

        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)
        # print('tokens:',t1)
        # print('t1_random:',t1_random)
        # print('t1_label:',t1_label)
        # print('\n')

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        
        # binary
        if(t3==self.class_name):
            label = 1
        else:
            label = 0

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "classification_label":label,
                  "ID":int(t0)}

        return {key: torch.tensor(value) for key, value in output.items()}

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1], self.lines[item][2], self.lines[item][3]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t0, t1, t2, t3 = line[:-1].split("\t")
            return t0, t1, t2, t3
