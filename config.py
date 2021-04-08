import torch
class Config(object):
    def __init__(self):
        self.emb_dim = 512  # dimension of word embeddings
        self.attention_dim = 512  # dimension of attention linear layers
        self.decoder_dim = 512  # dimension of decoder RNN
        self.dropout = 0.5
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
        #self.cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

        # Training parameters
        self.max_len= 5
        self.epochs = 100# number of epochs to train for (if early stopping is not triggered)
        self.epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
        self.batch_size = 128
        self.workers = 1  # for data-loading; right now, only 1 works with h5py
        self.encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
        self.decoder_lr = 4e-4  # learning rate for decoder
        self.grad_clip = 5.  # clip gradients at an absolute value of
        self.alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
        self.best_bleu4 = 0.  # BLEU-4 score right now
        self.print_freq = 100  # print training/validation stats every __ batches
        self.fine_tune_encoder = False  # fine-tune encoder?
        self.checkpoint = None  # path to checkpoint, None if none
        self.use_gpu = False
        self.dim_vid = 2048
        self.dim_hidden = 512
        self.self_crit_after = False
        self.self_crit_after = 5
        self.bidirectional = 1
        self.input_dropout_p = 0
        self.rnn_type = 'gru'
        self.rnn_dropout_p = 0.5
        self.learning_rate = 1e-3
        self.learning_rate_decay_every = 10
        self.weight_decay = 3e-3
        self.learning_rate_decay_rate = 1
        self.grad_clip = 5
        self.im_word = 256
        self.c3d_feat_dim = 400
