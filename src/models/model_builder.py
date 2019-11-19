import math
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class VaswaniPosEnc(nn.Module):
    """
    Generates a positional encoding as specified by Vaswani et al.

        PE_{pos, 2i} = sin(pos / 10000^{2i/dim})
        PE_{pos, 2i+1} = cos(pos / 10000^{2i/dim})

    In this notation, pos is the position and i is the dimension.

    The author's noted that sinusoidal positional encodings may help
    the model deal with sequences longer than those seen in training.
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(VaswaniPosEnc, self).__init__()
        self.dim = dim

        # Zero-initialized embedding vector and vector of position indices
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        # Scale value (10000^{2i / dim})
        # Here we do 10000^{2i/dim} = exp[2i * -(log(10000) / dim)]
        i2 = torch.arange(0, dim, 2, dtype=torch.float)
        scale =  torch.exp(i2 * -(math.log(10000.0) / dim))

        # Final embeddings based on even/odd i, buffered
        pe[:, 0::2] = torch.sin(position.float() * scale)
        pe[:, 1::2] = torch.cos(position.float() * scale)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # Droput as per Vaswani et al.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, step=None):
        # Multiply by sqrt(d_model) per Vaswani et al.
        _ = emb * math.sqrt(self.dim)

        # Sum original embedding with positional enc based on current step
        if (step):
            _ = _ + self.pe[:, step][:, None, :]
        else:
            _ = _ + self.pe[:, :emb.size(1)]

        return self.dropout(_)

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]



class Bert(nn.Module):
    """ Trivial wrapper around pretrained BERT model """
    EMBEDDING_SIZE = 768

    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask =mask)
        return encoded_layers[-1]



class Summarizer(nn.Module):

    def __init__(self, args, device, load_pretrained_bert = False, bert_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        d_model = self.bert.model.config.hidden_size

        # TODO if we try alternate positional embeddings, add switch to choose here
        self.pos_emb = VaswaniPosEnc(args.dropout, d_model)

        # Choose encoder based on runtime CLI flags
        if args.encoder == 'baseline':
            self.encoder = Summarizer._get_baseline_encoder(args, d_model)
        elif args.encoder == 'placeholder':
            raise NotImplementedError('havent implemented non-baseline model yet')
        else:
            raise ValueError('unknown encoder type %s' % args.encoder)

        # Final output dense layer mapping to probability via sigmoid
        self.dense = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Initialization strategy for weights
        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, ids, segs, clss, mask, mask_cls, sentence_range=None):
        """
        Forward pass for the summarizer

        Args:
            ids (tensor): input word ids for BERT (following wordpiece tokenization)
            segs (tensor): input segment ids for BERT, constructed as per BERTSUM
            clss (tensor): positions of [CLS] tokens
            mask (tensor): token level mask tensor
            mask_cls (tensor): sentence level mask tensor
        """
        # Run BERT to generate embeddings for each input token
        bert_emb = self.bert(ids, segs, mask)   # Batchx512x768

        # Extract and mask sentence-level embeddings
        # Sentence level embeddings are the embeddings produced by BERT for [CLS] tokens
        sents_vec = bert_emb[torch.arange(bert_emb.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        batch_size, n_sents = sents_vec.size(0), sents_vec.size(1)

        # Retreive positional encoding from lookup table, sum with sentence embeddings
        pos_emb = self.pos_emb.pe[:, :n_sents]
        _ = sents_vec * mask_cls[:, :, None].float()
        encoded_vec = _ + pos_emb   # batch_size x n_sents x 768
        assert encoded_vec.shape == (batch_size, n_sents, Bert.EMBEDDING_SIZE)

        # Reshape sent embeddings
        # Transformer expects input shape = n_sents x batch_size x 768
        encoded_vec = encoded_vec.transpose(0,1)
        assert encoded_vec.shape == (n_sents, batch_size, Bert.EMBEDDING_SIZE)

        # Reshape encoder mask input
        # Transformer expects mask shape = n_sents x n_sents
        # 0 -> unmasked, float('-inf') masked
        print(mask_cls)
        print(mask_cls[:,:,None])
        _ = torch.zeros_like(mask_cls, dtype=torch.float)
        _ = _.masked_fill(~mask_cls, float('-inf'))
        encoder_mask = _.expand(n_sents, -1)
        assert encoder_mask.shape == (n_sents, n_sents)

        # Pass through sentence level attention layers
        encoder_out = self.encoder(encoded_vec, mask=encoder_mask)
        assert encoder_out.shape == (n_sents, batch_size, Bert.EMBEDDING_SIZE)

        # Calculate sentence probabilities from final features
        sent_scores = self.sigmoid(self.dense(encoder_out)).squeeze(-1).transpose(0,1)
        assert sent_scores.shape == (1, n_sents)

        return sent_scores, mask_cls

    @staticmethod
    def _get_baseline_encoder(args, d_model):

        # A single attention encoder layer
        # Pytorch implementation follows Vaswani et al (w/ residuals, etc)
        encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model,
                nhead=args.heads,
                dim_feedforward=args.ff_size,
                dropout=args.dropout,
                activation='gelu')

        # The full encoder, composed of multiple encoder layers with Layernorm
        encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=args.inter_layers,
                norm=nn.LayerNorm(d_model, eps=1e-6))

        return encoder

    @staticmethod
    def _get_our_encoder(args, d_model):
        raise NotImplementedError('havent implemented our new encoder')
