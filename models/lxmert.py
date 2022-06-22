import torch
import torch.nn as nn
import numpy as np

import transformers

class LxmertVisualAnswerHead(nn.Module):
    def __init__(self, hid_dim=768, num_labels=28):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_labels),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)

class lxmert(nn.Module):
    def __init__(self, opt):
        super().__init__()
        num_labels = opt.num_ans_candidates
        hid_dim = opt.lxmert_hid_dim
        self.lxmert_encoder = transformers.LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.answer_head = LxmertVisualAnswerHead(hid_dim, num_labels)

    def forward(self,
                input_ids,
                attention_mask,
                visual_feats,
                visual_pos,
                token_type_ids,
                return_dict,
                output_attentions):
        output = self.lxmert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    visual_feats=visual_feats,
                                    visual_pos=visual_pos,
                                    token_type_ids=token_type_ids,
                                    return_dict=return_dict,
                                    output_attentions=output_attentions)
        result = self.answer_head(output['pooled_output'])
        return {'question_answering_score': result}