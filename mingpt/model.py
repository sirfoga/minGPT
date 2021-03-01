"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import numpy as np
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def apply_batchwise(func, M):
    tList = [ func(m) for m in torch.unbind(M, dim=0) ]  # batch is first index
    return torch.stack(tList, dim=0)


def minmax_norm():
    def _f(x):
        d = lambda x: torch.div(x, torch.max(x))
        return apply_batchwise(d, x)
    
    return _f


def normal(mu, sigma):
    def _f(x):
          return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
    return _f

  
def soft_quant(classes, sigma, norm=minmax_norm(), trans=True):
    basis = np.linspace(0, classes - 1, classes)
 
    def _f(x):
        N = normal(x * (classes - 1), sigma)
        vector = torch.stack([
            N(b) for b in basis
        ], dim=0)
        
        if norm:
            vector = norm(torch.transpose(vector, 0, 1))
            vector = torch.transpose(vector, 0, 1)
        
        if trans:
            vector = torch.transpose(vector, 0, 1)

        return vector
    return _f


def soft_torch(**kwargs):
    s = soft_quant(**kwargs)

    def _f(x):  # x ~ batch x elements
        return apply_batchwise(s, x)

    return _f


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    bert = False

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # 256 x 512
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))  # 1023 x 256
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd

        self.use_embd = config.use_embd  # else dense layer
        self.dense = nn.Linear(64, 256, bias=False)

        # self.ln_c = nn.LayerNorm((4, 1023))
        self.c = nn.Sequential(
             nn.Conv1d(1, 64, kernel_size=1, stride=1),
             nn.LeakyReLU(inplace=True),
             nn.Conv1d(64, 256, kernel_size=1, stride=1, bias=True),
        )
        self.n_soft_classes = 32
        self.gc = nn.Conv1d(self.n_soft_classes, 256, kernel_size=1, stride=1, bias=False)
        self.soft_q = soft_torch(classes=self.n_soft_classes, sigma=1, trans=False)

        # bert
        self.bert = config.bert

        # params
        # 
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.Conv1d, torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, plot_embd=False):
        b, t = idx.size()  # idx ~ batch x sequence length
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.bert:
            prob = 0.15

            M = torch.rand((b, t)).to('cuda')
            M = ( M > prob ).float()
            M = M.unsqueeze_(-1)
            M = M.repeat(1, 1, self.n_embd)

        # each index maps to a (learnable) vector
        if self.use_embd:
            token_embeddings = self.tok_emb(idx.long())
        else:
            token_embeddings = idx.float().view(b, t) 
            token_embeddings = self.soft_q(idx.to('cpu'))
            token_embeddings = self.gc(token_embeddings.to('cuda'))  # 4 x 256 x 1023

            # idx = idx.unsqueeze(-1).view([b, 1, t])  # 4 x 1 x 1023
            # token_embeddings = self.c(idx.float())  # 4 x 1023 * 256
            
            token_embeddings = token_embeddings.view(b, t, 256)

        _b, _t, _d = token_embeddings.size()

        if plot_embd:
            n_cols = 8
            n_rows = 40 // n_cols
            fig, axis = plt.subplots(n_rows, n_cols, figsize=(32, 16))
            for i, ax in enumerate(axis.ravel()):
                ti = token_embeddings[i]
                ti = ti.view(_t, _d).cpu().numpy()

                ax.imshow(ti, cmap='jet')
                ax.set_aspect(aspect=255.0/1023.0)

            plt.tight_layout()
            plt.savefig('./embds.png')
            plt.close('all')

        x = token_embeddings  # batch x t x n_embeddings
        if self.bert:
            x = x * M

        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = x + position_embeddings

        x = self.drop(x)  # dropout
        x = self.blocks(x)  # transformer
        x = self.ln_f(x)  # normalization on dense; this is `h` in original code ~ batch x t x n_embeddings

        # generative loss
        logits = self.head(x)  # batch x t x n_embeddings

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.long().view(-1))

            if self.bert:
                IM = 1.0 - M
                loss = loss * IM

        # finetuning loss (segmentation, in original paper they performed classification)
        # see https://github.com/milesial/Pytorch-UNet/tree/master/unet
        # with tf.variable_scope('clf', reuse=reuse):
        #     classes = shape_list(Y)[1]
        #     if hparams.clf:
        #         wclf = tf.get_variable('wclf', [classes, hparams.n_embd],
        #                               initializer=tf.random_normal_initializer(stddev=0.0))
        #     else:
        #         wclf = tf.zeros([classes, hparams.n_embd], dtype=tf.float32)

        # h = tf.reduce_mean(h, axis=1)  # average pool over sequence
        # clf_logits = tf.matmul(h, wclf, transpose_b=True)
        # clf_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=clf_logits, labels=Y)
        # results['clf_loss'] = tf.reduce_mean(clf_losses)

        return logits, loss
