import torch
from torch import nn
from config import Config
#from model import DecoderWithAttention, Encoder
from utils import *
from torch.utils.data import DataLoader
from model import DecoderRNN, EncoderRNN, S2VTAttModel
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

#from misc.rewards import get_self_critical_reward, init_cider_scorer
from model import DecoderRNN, EncoderRNN, S2VTAttModel
from ml_metrics import mapk


def val_map5(model, val_data, crit, config):
    fc_feats = val_data['fc_feats']
    labels = val_data['labels']
    masks = val_data['masks']
    model.eval()

    if config.use_gpu:
        torch.cuda.synchronize()
        fc_feats = fc_feats.cuda()
        labels = labels.cuda()
        masks = masks.cuda()

    with torch.no_grad():
        _, _, _, all_seq_preds = model(
            fc_feats, mode='inference', config = config)
        seq_probs, _, _, _ = model(fc_feats, config, labels, 'train')
    loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
    val_score = sum(
        [mapk(list(labels[:, i + 1].unsqueeze(1).cpu()), list(all_seq_preds[:, i, :].cpu()), 5)
         for i in range(3)]) / 3

    model.train()
    return loss, val_score
def train(train_loader, val_dataloader, crit, model,optimizer, lr_scheduler, config):
    model.train()
    # model = nn.DataParallel(model)

    for epoch in range(config.epochs):
        lr_scheduler.step()

        iteration = 0
        # If start self crit training
        #if config.self_crit_after and epoch >= config.self_crit_after:
         #   sc_flag = True
          #  init_cider_scorer()
        #else:
        sc_flag = False

        # batch size must > 117
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
           # if not sc_flag:

            #print(fc_feats.shape)

            seq_probs, _, _, _ = model(fc_feats, config, labels, 'train')
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            '''
            else:
                seq_probs, seq_preds = model(
                    fc_feats, mode='inference',config = config)
                reward = get_self_critical_reward(model, fc_feats, data,
                                                  seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds,
                               torch.from_numpy(reward).float().cuda())


'''
            loss.backward()
            clip_grad_value_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss = loss.item()
            if config.use_gpu:
                torch.cuda.synchronize()
            iteration += 1


            print("iter %d (epoch %d), train_loss = %.6f" %
                  (iteration, epoch, train_loss))
        #else:
         #   print("iter %d (epoch %d), avg_reward = %.6f" %
          #        (iteration, epoch, np.mean(reward[:, 0])))

        val_loss, val_score = val_map5(model, val_data, crit, config)
        print(f'val_loss:{val_loss} val_score:{val_score}')

    return model
'''
        if epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))
'''
def predict(model, dataset, vocab, config):
    object = json.load(open('./cs-5242-project-nus-2021-semester2/object1_object2.json', 'r'))
    relation = json.load(open('./cs-5242-project-nus-2021-semester2/relationship.json', 'r'))

    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    for data in loader:
        # forward the model to get loss
        fc_feats = data['fc_feats']

        if config.use_gpu:
            fc_feats = fc_feats.cuda()

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_prob, seq_preds, all_seq_logprobs, all_seq_preds = model(
                fc_feats,config = config, mode='inference' )
        answer,visualize_re = decode_index_into_final_answer(vocab, object, relation, all_seq_preds)


        answer_df = pd.DataFrame({'label': answer})
        #now_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())
        answer_df.to_csv('submission.csv', index_label='ID')

def main():
    """
    Training and validation.
    """
    config = Config()

    train_dataset = Load_data('train', config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = Load_data('val', config)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataset = Load_data('test', config)

    config.vocab_size = train_dataset.get_vocab_size()
    encoder = EncoderRNN(
            config.dim_vid + config.c3d_feat_dim,
            config.dim_hidden,
            bidirectional=config.bidirectional,
            input_dropout_p=config.input_dropout_p,
            rnn_cell=config.rnn_type,
            rnn_dropout_p=config.rnn_dropout_p,
        )
    decoder = DecoderRNN(
            config.vocab_size,
            config.max_len,
            config.dim_hidden,
            config.im_word,
            input_dropout_p=config.input_dropout_p,
            rnn_cell=config.rnn_type,
            rnn_dropout_p=config.rnn_dropout_p,
            bidirectional=config.bidirectional,
            using_gpu=config.use_gpu)
    model = S2VTAttModel(encoder, decoder)

    if config.use_gpu == True:
        model = model.cuda()
    crit = LanguageModelCriterion()
    #rl_crit = utils.RewardCriterion()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.learning_rate_decay_every,
        gamma=config.learning_rate_decay_rate)

    model = train(train_dataloader, val_dataloader, crit, model, optimizer, exp_lr_scheduler, config)

    predict(model, test_dataset, test_dataset.get_vocab(), config)

if __name__ == '__main__':
    main()