import math
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from onmt.utils.logging import logger, init_logger
from onmt.utils import use_gpu, Optimizer

model_flags = [
    'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval'
]


def load_model(args, device, load_bert=False, checkpoint=None):
    """Loads a model given by args.encoder using a checkpoint file if specified"""
    config = BertConfig.from_json_file(args.bert_config_path)

    if args.encoder == 'bertsum':
        logger.info('Loading BertSum baseline model')
        model = BertSum(args, device, load_bert=load_bert, bert_config=config)
    elif args.encoder == 'bertsum2':
        logger.info('Loading BertSum bi-transformer model')
        model = BertSumBitransform(args,
                                   device,
                                   load_bert=load_bert,
                                   bert_config=config)
    elif args.encoder == 'bertsum_conv':
        logger.info('Loading BertSum conv model')
        model = BertSumConv(args,
                            device,
                            load_bert=load_bert,
                            bert_config=config)
    else:
        raise ValueError('unknown encoder %s' % args.encoder)

    # Load checkpointed model
    if checkpoint:
        logger.info('Loading checkpoint from %s' % checkpoint)
        checkpoint = torch.load(checkpoint,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        model.load_cp(checkpoint)

    # Load optimizer if resuming training, otherwise fresh optimizer
    if checkpoint and args.mode == 'train':
        logger.info("Restoring checkpointed optimizer state")
        #optim = Optimizer.from_opt(model, args, checkpoint)
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.set_parameters(list(model.named_parameters()))
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)

        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    else:
        logger.info("Using fresh optimizer state")
        optim = Optimizer.from_opt(model, args)

    return model, optim


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
        enc = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        # Scale value (10000^{2i / dim})
        # Here we do 10000^{2i/dim} = exp[2i * -(log(10000) / dim)]
        i2 = torch.arange(0, dim, 2, dtype=torch.float)
        scale = torch.exp(i2 * -(math.log(10000.0) / dim))

        # Sin at even positions
        enc[:, 0::2] = torch.sin(position.float() * scale)

        # Cos at odd positions
        enc[:, 1::2] = torch.cos(position.float() * scale)

        enc = enc.unsqueeze(0)
        self.register_buffer('enc', enc)
        # Droput as per Vaswani et al.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, step=None):
        # Multiply by sqrt(d_model) per Vaswani et al.
        _ = emb * math.sqrt(self.dim)

        # Sum original embedding with positional enc based on current step
        if (step):
            _ = _ + self.enc[:, step][:, None, :]
        else:
            _ = _ + self.enc[:, :emb.size(1)]

        return self.dropout(_)


class BertSum(nn.Module):

    EMBEDDING_SIZE = 768

    def __init__(self, args, device, load_bert=False, bert_config=None):
        super(BertSum, self).__init__()
        self.args = args
        self.device = device

        if load_bert:
            self.bert = BertModel.from_pretrained('bert-base-uncased',
                                                  cache_dir=args.temp_dir)
        else:
            self.bert = BertModel(bert_config)

        d_model = self.bert.config.hidden_size

        # TODO if we try alternate positional embeddings, add switch to choose here
        self.pos_emb = VaswaniPosEnc(args.dropout, d_model)

        # A single attention encoder layer
        # Pytorch implementation follows Vaswani et al (w/ residuals, etc)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead=args.heads,
            dim_feedforward=args.ff_size,
            dropout=args.dropout,
            activation='gelu')

        # The full encoder, composed of multiple encoder layers with Layernorm
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.inter_layers,
            norm=nn.LayerNorm(d_model, eps=1e-6))

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
        self.load_state_dict(pt['model'], strict=False)

    def forward(self, ids, segs, clss, mask, mask_cls, step=None):
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
        bert_emb, _ = self.bert(ids, segs, attention_mask=mask)
        bert_emb = bert_emb[-1]  # batch_size x 512 x768

        # Extract and mask sentence-level embeddings
        # Sentence level embeddings are the embeddings produced by BERT for [CLS] tokens
        sents_vec = bert_emb[torch.arange(bert_emb.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        batch_size, n_sents = sents_vec.size(0), sents_vec.size(1)

        # Retreive positional encoding from lookup table, sum with sentence embeddings
        #pos_emb = self.pos_emb.pe[:, :n_sents]
        pos_emb = self.pos_emb(sents_vec)
        _ = sents_vec * mask_cls[:, :, None].float()
        encoded_vec = _ + pos_emb  # batch_size x n_sents x 768
        assert encoded_vec.shape == (batch_size, n_sents,
                                     BertSum.EMBEDDING_SIZE)

        # Reshape sent embeddings
        # Transformer expects input shape = n_sents x batch_size x 768
        encoded_vec = encoded_vec.transpose(0, 1)

        # Pass through sentence level attention layers
        encoder_out = self.encoder(encoded_vec, src_key_padding_mask=~mask_cls)
        assert encoder_out.shape == (n_sents, batch_size,
                                     BertSum.EMBEDDING_SIZE)

        # Calculate sentence probabilities from final features
        sent_scores = self.sigmoid(
            self.dense(encoder_out)).squeeze(-1).transpose(0, 1)
        assert sent_scores.shape == (batch_size, n_sents)

        return sent_scores, mask_cls


class BertSumBitransform(BertSum):

    EMBEDDING_SIZE = 768

    def __init__(self, args, device, load_bert=False, bert_config=None):
        super().__init__(args,
                         device,
                         load_bert=load_bert,
                         bert_config=bert_config)

        d_model = self.bert.config.hidden_size

        # Second encoder for token level embeddings
        encoder_l_2 = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead=args.heads,
            dim_feedforward=args.ff_size,
            dropout=args.dropout,
            activation='gelu')

        # The full token level encoder
        self.encoder2 = torch.nn.TransformerEncoder(
            encoder_l_2,
            num_layers=args.inter_layers,
            norm=nn.LayerNorm(d_model, eps=1e-6))

        # Initialization strategy for weights
        if args.param_init != 0.0:
            for p in self.encoder2.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder2.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=False)

    def forward(self, ids, segs, clss, mask, mask_cls, step=None):
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
        bert_emb, _ = self.bert(ids, segs, attention_mask=mask)
        bert_emb = bert_emb[-1]  # batch_size x 512 x768

        # Hack together embeddings that are the sum of BERT token embeddings
        # on a per sentence basis
        _vals = torch.arange(len(mask[0])).to(clss)
        _vals = _vals.unsqueeze(0).unsqueeze(0)
        _vals = _vals.repeat(len(clss), len(clss[0]), 1)
        _ = torch.full(size=(len(clss), 1), fill_value=513).to(clss)
        _ = torch.cat((clss, _), dim=-1)
        _ = _.unsqueeze(-1).repeat(1, 1, len(mask[0]))
        _ = _vals.ge(_[:, :-1]) & _vals.lt(_[:, 1:])
        _ = _.unsqueeze(-1).repeat(1, 1, 1, BertSum.EMBEDDING_SIZE)
        expand_bert = bert_emb.unsqueeze(1).repeat(1, len(mask_cls[0]), 1, 1)
        _embeddings = torch.zeros(_.shape).to(bert_emb)
        _embeddings = _embeddings.masked_scatter(_, expand_bert)
        sum_token_embeddings = _embeddings.sum(dim=2)

        # Extract and mask sentence-level embeddings
        # Sentence level embeddings are the embeddings produced by BERT for [CLS] tokens
        sents_vec = bert_emb[torch.arange(bert_emb.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        assert sum_token_embeddings.shape == sents_vec.shape

        batch_size, n_sents = sents_vec.size(0), sents_vec.size(1)

        # Retreive positional encoding from lookup table, sum with sentence embeddings
        #pos_emb = self.pos_emb.pe[:, :n_sents]
        pos_emb = self.pos_emb(sents_vec)
        _ = sents_vec * mask_cls[:, :, None].float()
        encoded_vec = _ + pos_emb  # batch_size x n_sents x 768
        assert encoded_vec.shape == (batch_size, n_sents,
                                     BertSum.EMBEDDING_SIZE)

        # Reshape sent embeddings
        # Transformer expects input shape = n_sents x batch_size x 768
        encoded_vec = encoded_vec.transpose(0, 1)
        assert encoded_vec.shape == (n_sents, batch_size,
                                     BertSum.EMBEDDING_SIZE)

        # Pass through sentence level attention layers
        encoder_out = self.encoder(encoded_vec, src_key_padding_mask=~mask_cls)
        assert encoder_out.shape == (n_sents, batch_size,
                                     BertSum.EMBEDDING_SIZE)

        sum_token_embeddings = sum_token_embeddings.transpose(0, 1)
        assert sum_token_embeddings.shape == (n_sents, batch_size,
                                              BertSum.EMBEDDING_SIZE)

        encoder2_out = self.encoder2(sum_token_embeddings,
                                     src_key_padding_mask=~mask_cls)

        join = encoder_out + encoder2_out

        # Calculate sentence probabilities from final features
        sent_scores = self.sigmoid(self.dense(join)).squeeze(-1).transpose(
            0, 1)
        assert sent_scores.shape == (batch_size, n_sents)

        return sent_scores, mask_cls


class BertSumConv(nn.Module):

    EMBEDDING_SIZE = 768

    def __init__(self, args, device, load_bert=False, bert_config=None):
        super(BertSumConv, self).__init__()
        self.args = args
        self.device = device

        if load_bert:
            self.bert = BertModel.from_pretrained('bert-base-uncased',
                                                  cache_dir=args.temp_dir)
        else:
            self.bert = BertModel(bert_config)

        channel_in = self.bert.config.hidden_size
        d_model = channel_in // 2
        self.d_model = d_model
        kernel = 3
        self.conv = torch.nn.Conv1d(channel_in,
                                    d_model,
                                    kernel,
                                    1,
                                    bias=True,
                                    padding=1)
        self.conv_act = torch.nn.ReLU()

        # TODO if we try alternate positional embeddings, add switch to choose here
        self.pos_emb = VaswaniPosEnc(args.dropout, channel_in)

        # A single attention encoder layer
        # Pytorch implementation follows Vaswani et al (w/ residuals, etc)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead=args.heads,
            dim_feedforward=args.ff_size,
            dropout=args.dropout,
            activation='gelu')

        # The full encoder, composed of multiple encoder layers with Layernorm
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.inter_layers,
            norm=nn.LayerNorm(d_model, eps=1e-6))

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
        self.load_state_dict(pt['model'], strict=False)

    def forward(self, ids, segs, clss, mask, mask_cls, step=None):
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
        bert_emb, _ = self.bert(ids, segs, attention_mask=mask)
        bert_emb = bert_emb[-1]  # batch_size x 512 x768

        # Extract and mask sentence-level embeddings
        # Sentence level embeddings are the embeddings produced by BERT for [CLS] tokens
        sents_vec = bert_emb[torch.arange(bert_emb.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        batch_size, n_sents = sents_vec.size(0), sents_vec.size(1)

        # Retreive positional encoding from lookup table, sum with sentence embeddings
        #pos_emb = self.pos_emb.pe[:, :n_sents]
        pos_emb = self.pos_emb(sents_vec)
        _ = sents_vec * mask_cls[:, :, None].float()
        encoded_vec = _ + pos_emb  # batch_size x n_sents x 768
        assert encoded_vec.shape == (batch_size, n_sents,
                                     BertSum.EMBEDDING_SIZE)

        encoded_vec = encoded_vec.transpose(1, 2)  # conv expects N x C x L
        _ = self.conv(encoded_vec)
        _ = self.conv_act(_)

        # Transformer expects input shape = n_sents x batch_size x 768
        _ = _.transpose(0, 1)
        _ = _.transpose(0, 2)

        # Pass through sentence level attention layers
        encoder_out = self.encoder(_, src_key_padding_mask=~mask_cls)

        # Calculate sentence probabilities from final features
        sent_scores = self.sigmoid(
            self.dense(encoder_out)).squeeze(-1).transpose(0, 1)
        assert sent_scores.shape == (batch_size, n_sents)

        return sent_scores, mask_cls
