import torch
from torch import nn
from config import Config
from utils import *
from torch.utils.data import DataLoader
from model import Decoder, Encoder, GRUattModel
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torchviz import make_dot
import hiddenlayer as h
from ml_metrics import mapk


def train(train_loader, val_dataloader, model, optimizer, lr_scheduler, config):
    model.train()
    crit = LanguageModelCriterion()
    for epoch in range(config.epochs):
        lr_scheduler.step()
        iteration = 0
        val_data = None
        for data in val_dataloader:
            val_data = data
        for data in train_loader:
            fc_feats = data['fc_feats']
            labels = data['labels']
            masks = data['masks']

            if config.use_gpu:
                torch.cuda.synchronize()
                fc_feats = fc_feats.cuda()
                labels = labels.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()

            seq_probs, _, _, _ = model(fc_feats, config, labels, 'train')
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])

            loss.backward()
            clip_grad_value_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss = loss.item()
            iteration += 1
            print("iter %d (epoch %d), train_loss = %.6f" %(iteration, epoch, train_loss))

        torch.save(model.state_dict(), '%d.pth'%epoch)
        val_loss, val_score = validation(model, val_data, crit, config)
        print(f'val_loss:{val_loss} val_score:{val_score}')

    return model

def validation(model, val_data, crit, config):
    fc_feats = val_data['fc_feats']
    labels = val_data['labels']
    masks = val_data['masks']
    model.eval()
    with torch.no_grad():
        _, _, _, all_seq_preds = model(fc_feats, mode='inference', config=config)
        seq_probs, _, _, _ = model(fc_feats, config, labels, 'train')
    loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
    val_score = sum([mapk(list(labels[:, i + 1].unsqueeze(1)), list(all_seq_preds[:, i, :]), 5) for i in range(3)]) / 3
    model.train()
    return loss, val_score

def predict(model, dataset, vocab, config):
    object = json.load(open('./cs-5242-project-nus-2021-semester2/object1_object2.json', 'r'))
    relation = json.load(open('./cs-5242-project-nus-2021-semester2/relationship.json', 'r'))

    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    for data in loader:
        fc_feats = data['fc_feats']

        if config.use_gpu:
            fc_feats = fc_feats.cuda()

        with torch.no_grad():
            seq_prob, seq_preds, all_seq_logprobs, all_seq_preds = model(fc_feats, config=config, mode='inference')
        res = decode_index_into_final_answer(vocab, object, relation, all_seq_preds)
        res = pd.DataFrame({'label': res})

        res.to_csv('submission.csv', index_label='ID')

def main():

    config = Config()
#load data
    train_dataset = Load_data('train', config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = Load_data('val', config)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataset = Load_data('test', config)

    config.vocab_size = train_dataset.get_vocab_size()
    #model
    encoder = Encoder(config.dim_vid, config.dim_hidden, input_dropout_p=config.input_dropout_p, rnn_cell=config.rnn_type,
                      rnn_dropout_p=config.rnn_dropout_p,)
    decoder = Decoder(config.vocab_size,  config.max_len, config.dim_hidden, config.im_word, input_dropout_p=config.input_dropout_p,
            rnn_dropout_p=config.rnn_dropout_p, bidirectional=config.bidirectional, using_gpu=config.use_gpu)

    model = GRUattModel(encoder, decoder)
    if config.load_checkpoint:
        model.load_state_dict(torch.load('86.pth'))
        predict(model, test_dataset, test_dataset.get_vocab(), config)
        exit()
    if config.use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    lr_scheduler = optim.lr_scheduler.StepLR( optimizer, step_size=config.learning_rate_decay_every, gamma=config.learning_rate_decay_rate)

    model = train(train_dataloader, val_dataloader, model, optimizer, lr_scheduler, config)

    predict(model, test_dataset, test_dataset.get_vocab(), config)

if __name__ == '__main__':
    main()