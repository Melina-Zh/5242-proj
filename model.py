import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
        # self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1,
                 using_gpu=True):
        super(DecoderRNN, self).__init__()

        self.using_gpu = using_gpu

        self.bidirectional_encoder = bidirectional

        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output, dim_word)
        self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden + dim_word,
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                config=None,
                targets=None,
                mode='inference'
                ):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        sample_max = 1
        beam_size = 1
        temperature = 1.0

        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        seq_logprobs = []
        seq_preds = []
        all_seq_logprobs = []
        all_seq_preds = []
        self.rnn.flatten_parameters()
        if mode == 'train':
            # use targets as rnn inputs
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

            #if beam_size > 1:
             #   return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.max_length - 1):
                context = self.attention(
                    decoder_hidden.squeeze(0), encoder_outputs)

                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size)
                    all_it = torch.LongTensor([[self.sos_id] * 119 for _ in range(batch_size)])
                    if self.using_gpu:
                        it = it.cuda()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    top5Logprobs, all_it = torch.topk(logprobs, k=119, largest=True, sorted=True)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    all_seq_logprobs += [top5Logprobs.unsqueeze(1)]
                    it = it.view(-1).long()
                    all_it = all_it.long()

                else:
                    # sample according to distribuition
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    if self.using_gpu:
                        it = it.cuda()
                    sampleLogprobs = logprobs.gather(1, it)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()

                seq_preds.append(it.view(-1, 1))
                all_seq_preds += [all_it.unsqueeze(1)]

                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds[1:], 1)
            all_seq_logprobs = torch.cat(all_seq_logprobs, 1)
            all_seq_preds = torch.cat(all_seq_preds[1:], 1)

        return seq_logprobs, seq_preds, all_seq_logprobs,all_seq_preds

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



class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = True if bidirectional else False
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                 bidirectional=self.bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden


class S2VTAttModel(nn.Module):
    def __init__(self, encoder, decoder):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, config=None, target_variable=None,
                mode='inference' ):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds,all_seq_logprobs,all_seq_preds = self.decoder(encoder_outputs, encoder_hidden, config, target_variable, mode)
        return seq_prob, seq_preds,all_seq_logprobs,all_seq_preds

