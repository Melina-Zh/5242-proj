import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):

        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, self.dim * 2)

        aw = F.tanh(self.linear1(inputs))
        o = self.linear2(aw)
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

class Decoder(nn.Module):

    def __init__(self,vocab_size,max_len, dim_hidden, dim_word, n_layers=1, bidirectional=False, input_dropout_p=0.1,
                 rnn_dropout_p=0.1, using_gpu=True):
        super(Decoder, self).__init__()

        self.using_gpu = using_gpu
        self.bidirectional_encoder = bidirectional
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2
        self.dim_word = dim_word
        self.max_length = max_len
        self.bos_id = 1
        self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output, dim_word)
        self.attention = Attention(self.dim_hidden)
        self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(self.dim_hidden + dim_word,self.dim_hidden, n_layers, batch_first=True, dropout=rnn_dropout_p)
        self.out = nn.Linear(self.dim_hidden, self.dim_output)
        self.testsize = 119
        self._init_weights()

    def forward(self, encoder_outputs, encoder_hidden, config=None, targets=None, mode='inference'):

        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        seq_logprobs = []
        seq_preds = []
        all_seq_logprobs = []
        all_seq_preds = []
        self.rnn.flatten_parameters()
        if mode == 'train':
            targets_emb = self.embedding(targets)
            for i in range(self.max_length - 1):
                current_words = targets_emb[:, i, :]
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            seq_logprobs = torch.cat(seq_logprobs, 1)

        elif mode == 'inference':
            context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
            it = torch.LongTensor([self.bos_id] * batch_size)
            all_it = torch.LongTensor([[self.bos_id] * self.testsize for _ in range(batch_size)])
            seq_preds.append(it.view(-1, 1))
            all_seq_preds += [all_it.unsqueeze(1)]

            xt = self.embedding(it)
            decoder_input = torch.cat([xt, context], dim=1)
            decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            logprobs = F.log_softmax(self.out(decoder_output.squeeze(1)), dim=1)

            for t in range(1, self.max_length - 1):
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                sampleLogprobs, it = torch.max(logprobs, 1)
                top5Logprobs, all_it = torch.topk(logprobs, k=119, largest=True, sorted=True)
                seq_logprobs.append(sampleLogprobs.view(-1, 1))
                all_seq_logprobs += [top5Logprobs.unsqueeze(1)]
                it = it.view(-1).long()
                all_it = all_it.long()

                seq_preds.append(it.view(-1, 1))
                all_seq_preds += [all_it.unsqueeze(1)]

                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                logprobs = F.log_softmax(self.out(decoder_output.squeeze(1)), dim=1)
#add eos
            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds[1:], 1)
            all_seq_logprobs = torch.cat(all_seq_logprobs, 1)
            all_seq_preds = torch.cat(all_seq_preds[1:], 1)

        return seq_logprobs, seq_preds, all_seq_logprobs, all_seq_preds

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


class Encoder(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, rnn_cell='gru'):
        super(Encoder, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = True
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                 bidirectional=self.bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):

        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden


class GRUattModel(nn.Module):
    def __init__(self, encoder, decoder):

        super(GRUattModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, config=None, target_variable=None,
                mode='inference'):

        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds,all_seq_logprobs,all_seq_preds = self.decoder(encoder_outputs, encoder_hidden, config, target_variable, mode)
        return seq_prob, seq_preds,all_seq_logprobs,all_seq_preds

