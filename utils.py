import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class Load_data(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, mode, config):
        super(Load_data, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open('./cs-5242-project-nus-2021-semester2/5242_data/train_val_label.json'))
        info = json.load(open('./cs-5242-project-nus-2021-semester2/5242_data/info.json'))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = './cs-5242-project-nus-2021-semester2/resnet152/'
        #self.c3d_feats_dir = opt['c3d_feats_dir']
        #self.with_c3d = opt['with_c3d']
        #print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = config.max_len
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])

        fc_feat = []
        #print(self.feats_dir)
        #for dir in self.feats_dir:
         #   print(dir)
        fc_feat = np.load(os.path.join(self.feats_dir, 'video%i.npy' % (ix)))
        #fc_feat = np.concatenate(fc_feat, axis=1)
        label = np.zeros(self.max_len)
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

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = 'video%i' % (ix)
        return data

    def __len__(self):
        return len(self.splits[self.mode])