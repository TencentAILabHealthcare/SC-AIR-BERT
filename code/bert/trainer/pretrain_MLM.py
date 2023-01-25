import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import BERTIR, BERT
from .optim_schedule import ScheduledOptim

import tqdm

class BERTTrainer_MLM:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTIR(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        return self.iteration(epoch, self.train_data)

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        # suxiaona
        total_mask_correct = 0
        total_mask_element=0

        loss_list = []
        mask_acc_list = []

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
           

            # 1. forward the next_sentence_prediction and masked_lm model
            # data["bert_input"].shape = [batch_size,seq_len]
            mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
            
            # print('bert_input:',data['bert_input'].shape)
            # print('bert_label:',data["bert_label"])
            # print('pred:',mask_lm_output.shape)
            
            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # suxiaona mask correct
            avg_loss += loss.item()
            correct_mask = ((mask_lm_output.argmax(dim=-1).eq(data["bert_label"]))&(data["bert_label"]!=0)).sum().item()
            total_mask_correct += correct_mask
            total_mask_element += (data["bert_label"]!=0).sum().item()

            # print("\n")
            # print('correct:',correct_mask)
            # print('total:',(data["bert_label"]!=0).sum().item())

            # suxiaona
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                'loss':loss.item(),
                "mask_avg_acc": total_mask_correct/total_mask_element * 100,
                'mask_acc':correct_mask/(data["bert_label"]!=0).sum().item()*100
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

            loss_list.append(post_fix['avg_loss'])            
            mask_acc_list.append(post_fix['mask_avg_acc'])  

            epoch_loss = avg_loss / len(data_iter)
            epoch_mask_acc = total_mask_correct/total_mask_element * 100

            history = {
                'loss':loss_list,
                'acc':mask_acc_list,
                'epoch_loss':epoch_loss,
                'epoch_acc':epoch_mask_acc
            }
            # and post_fix['iter']<20
            # if(train==True and post_fix['epoch'] == 4):
            #     temp = str(post_fix['iter'])
            #     path = '/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/result/10x/ab/output/'
            #     torch.save(data['bert_input'],path + 'input_'+temp+'.pth')
            #     torch.save(data['bert_label'],path + 'lable_'+temp+'.pth')
            #     torch.save(mask_lm_output.argmax(dim=-1),path + 'output_'+temp+'.pth')
            # torch.save(mask_lm_output.argmin(dim=-1),'/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST/result/10x_RAKFKQLL_TCR/output_min.pth')

            # suxiaona
        print("EP%d_%s" % (epoch, str_code), "avg_loss:" ,avg_loss / len(data_iter),
              'mask_avg_acc=', total_mask_correct/total_mask_element * 100)
        
        return history

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
