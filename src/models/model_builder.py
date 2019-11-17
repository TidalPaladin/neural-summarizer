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


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask =mask)
        top_vec = encoded_layers[-1]
        return top_vec



class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, bert_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        d_model = self.bert.model.config.hidden_size

        self.pos_emb = PositionalEncoding(args.dropout, d_model)

        # Choose encoder based on runtime CLI flags
        if args.encoder == 'baseline':
            self.encoder = Summarizer._get_baseline_encoder(args, d_model)
        elif args.encoder == 'placeholder':
            raise NotImplementedError('havent implemented non-baseline model yet')
        else:
            raise ValueError('unknown encoder type %s' % args.encoder)

        # Final output dense layer mapping to probability
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

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        # Sum positional encoding with sent vec
        batch_size, n_sents = sents_vec.size(0), sents_vec.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        _ = sents_vec * mask_cls[:, :, None].float()
        encoded_vec = _ + pos_emb

        sent_mask = ~mask_cls[:,1]
        encoder_out = self.encoder(encoded_vec, mask=sent_mask)
        sent_scores = self.sigmoid(self.dense(encoder_out)).squeeze(-1) * mask_cls.float()
        return sent_scores, mask_cls

    @staticmethod
    def _get_baseline_encoder(args, d_model):
        # A single attention encoder layer
        print(args.ff_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model,
                nhead=args.heads,
                dim_feedforward=args.ff_size,
                dropout=args.dropout,
                activation='gelu')

        # The encoder, composed of multiple encoder layers with Layernorm
        encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=args.inter_layers,
                norm=nn.LayerNorm(d_model, eps=1e-6))

        return encoder
