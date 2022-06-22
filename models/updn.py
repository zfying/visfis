import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch
import timeit

from components.language_model import WordEmbedding, QuestionEmbedding, WordEmbedding_ramen, UpDnQuestionEmbedding_ramen
from components.classifier import SimpleClassifier, SimpleClassifier_ramen
from components.fc import FCNet, GTH, FCNet_ramen
from components.attention import Att_3, UpDnAttention_ramen


class UpDn(nn.Module):
    def __init__(self, opt):
        super(UpDn, self).__init__()
        num_hid = opt.num_hid
        activation = opt.activation
        dropG = opt.dropG
        dropW = opt.dropW
        dropout = opt.dropout
        dropL = opt.dropL
        norm = opt.norm
        dropC = opt.dropC
        self.opt = opt
        
        ### MYCODE ###
        full_v_dim = opt.full_v_dim
        ### END MYCODE ###
        
        print(f"ntokens {opt.ntokens}")
        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=dropW)
        self.w_emb.init_embedding(f'{opt.data_dir}/glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid)

        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=dropL, norm=norm, act=activation)
        self.gv_net = FCNet([full_v_dim, num_hid], dropout=dropL, norm=norm, act=activation)

        self.gv_att_1 = Att_3(v_dim=full_v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.gv_att_2 = Att_3(v_dim=full_v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=opt.num_ans_candidates, dropout=dropC, norm=norm, act=activation)

    def forward(self, q, gv):

        """Forward
        q: [batch_size, seq_length]
        c: [batch, 5, 20]
        return: logits, not probs
        """

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]

        att_1 = self.gv_att_1(gv, q_emb)  # [batch, 1, v_dim]
        att_2 = self.gv_att_2(gv, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2
        gv_embs = (att_gv * gv)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        q_repr = self.q_net(q_emb)
        gv_repr = self.gv_net(gv_emb)
        joint_repr = q_repr * gv_repr

        logits = self.classifier(joint_repr)
        ansidx = torch.argsort(logits, dim=1, descending=True)

        return w_emb, logits, att_gv,  ansidx

    
    
    
class UpDn_ramen_new(nn.Module):
    def __init__(self, config):
        super(UpDn_ramen_new, self).__init__()
        
        full_v_dim = config.full_v_dim
        self.full_v_dim = full_v_dim
        
        print(f"ntokens {config.ntokens}")
        self.w_emb = WordEmbedding_ramen(config.ntokens, 300, 0.0)
        # self.w_emb.init_embedding(f'{config.data_dir}/glove6b_init_300d.npy')
        self.q_emb = UpDnQuestionEmbedding_ramen(300, config.num_hid, 1, False, 0.0)
        self.v_att = UpDnAttention_ramen(full_v_dim, self.q_emb.num_hid, config.num_hid)
        self.q_net = FCNet_ramen([self.q_emb.num_hid, config.num_hid])
        self.v_net = FCNet_ramen([config.full_v_dim, config.num_hid])
        self.classifier = SimpleClassifier_ramen(
            config.num_hid, config.num_hid * 2, config.num_ans_candidates, 0.5)

    def forward(self, q, v):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        ansidx = torch.argsort(logits, dim=1, descending=True)
        
        return w_emb, logits, None, ansidx
    
    
class UpDn_ramen(nn.Module):
    def __init__(self, opt):
        super(UpDn_ramen, self).__init__()
        num_hid = opt.num_hid
        self.opt = opt
        
        ### MYCODE ###
        self.full_v_dim = opt.full_v_dim
        # self.oracle_type = opt.oracle_type
        # self.oracle_threshold = opt.oracle_threshold
        # if self.oracle_type == 'wordvec':
        #     self.oracle_embed = nn.Embedding(2, opt.oracle_embed_size, padding_idx=-1)  # Binary
        ### END MYCODE ###
        
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
        ### MYCODE ###
        # if self.oracle_type == 'wordvec':
        #     visual_features, hint_scores = gv[: , : , :-1], gv[: , : , -1]
        #     # apply mask
        #     oracle_mask = (hint_scores > self.oracle_threshold).long()
        #     oracle_embedding = self.oracle_embed(oracle_mask)
        #     gv =torch.cat([visual_features, oracle_embedding], dim=2) 
        ### END MYCODE ###
        
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