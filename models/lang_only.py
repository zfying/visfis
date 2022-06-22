import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from components.language_model import WordEmbedding, QuestionEmbedding
from components.classifier import SimpleClassifier, PaperClassifier
from components.fc import FCNet, GTH
from components.attention import UpDnAttention_ramen

import torch
import timeit


class UpDn_ramen(nn.Module):
    def __init__(self, opt):
        super(UpDn_ramen, self).__init__()
        num_hid = opt.num_hid
        self.opt = opt
        
        self.full_v_dim = opt.full_v_dim
        
        print(f"ntokens {opt.ntokens}")
        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=0.0)
        self.w_emb.init_embedding(f'{opt.data_dir}/glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid)

        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=0.0)
        self.gv_net = FCNet([self.full_v_dim, num_hid], dropout=0.0)

        self.gv_att = UpDnAttention_ramen(v_dim=self.full_v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=0.2)
        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=opt.num_ans_candidates, dropout=0.5)

    def forward(self, q, gv):

        """Forward
        q: [batch_size, seq_length]
        c: [batch, 5, 20]
        return: logits, not probs
        """
        # set visual features to zeros
        gv = torch.zeros(gv.size()).cuda()
        # q embedding
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        
        # att embedding
        att_gv = self.gv_att(gv, q_emb)  # [batch, 1, v_dim]
        gv_emb = (att_gv * gv).sum(1)
        
        # FCN
        q_repr = self.q_net(q_emb)
        gv_repr = self.gv_net(gv_emb)
        joint_repr = q_repr * gv_repr
        
        # classify
        logits = self.classifier(joint_repr)
        ansidx = torch.argsort(logits, dim=1, descending=True)

        return w_emb, logits, att_gv,  ansidx