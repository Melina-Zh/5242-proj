import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


class Load_data(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, mode, config):
        super(Load_data, self).__init__()
        self.mode = mode

        self.captions = json.load(open('./cs-5242-project-nus-2021-semester2/5242_data/train_val_label.json'))
        info = json.load(open('./cs-5242-project-nus-2021-semester2/5242_data/info.json'))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']

        self.splits = info['videos']


        self.feats_dir = './cs-5242-project-nus-2021-semester2/resnet152/'
        self.c3d_feats_dir = './cs-5242-project-nus-2021-semester2/conv3d_feats/'

        self.max_len = config.max_len


    def __getitem__(self, ix):

        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])


        fc_feat = np.load(os.path.join(self.feats_dir, 'video%i.npy' % (ix)))
        #c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy' % (ix)))
        #fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)

        mask = np.zeros(self.max_len)

        # test data
        if f'video{ix}' not in self.captions:
            data = {}
            data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
            data['video_ids'] = 'video%i' % (ix)
            return data

        captions = self.captions[f'video{ix}']['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]


        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['video_ids'] = 'video%i' % (ix)
        return data

    def __len__(self):
        return len(self.splits[self.mode])


def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


def decode_index_into_final_answer(ix_to_word, object_ix, relation_ix, seq):
    res = []
    n = seq.shape[0]
    m = seq.shape[2]
    for k in range(n):
        left_obj, right_obj, relation = [], [], []
        for i in range(m):
            word1 = ix_to_word[str(seq[k, 0, i].item())]
            if len(left_obj) < 5 and word1 in object_ix:
                left_obj += [object_ix[word1]]
            word2 = ix_to_word[str(seq[k, 1, i].item())]
            if len(relation) < 5 and word2 in relation_ix:
                relation += [relation_ix[word2]]
            word3 = ix_to_word[str(seq[k, 2, i].item())]
            if len(right_obj) < 5 and word3 in object_ix:
                right_obj += [object_ix[word3]]

        res += [list2str(left_obj), list2str(relation), list2str(right_obj)]

    return res

def list2str(arr:[int]) -> str:
    re = ''
    for num in arr:
        re += f'{num} '
    return re[:-1]

class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):


        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output
