import logging
from typing import List

import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class BERT(BertPreTrainedModel):
    """Bert model"""

    def __init__(self, config):
        super(BERT, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_attentions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_attentions=output_attentions)  # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class MultitaskBertClassificationModel(nn.Module):
    """Multitask Text classification model class using BERT model"""

    def __init__(
            self,
            encoder: BERT,
            src_pad_idx: int,
            out_dims: List[int],
            fc_dim: int,
    ):
        super().__init__()

        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.out_dims = out_dims
        self.fc_dim = fc_dim
        self.make_src_mask = self.make_src_mask_bert
        self.fc = nn.Linear(self.encoder.config.hidden_size, fc_dim)
        self.clfs = nn.ModuleList([nn.Linear(fc_dim, out_dim) for out_dim in self.out_dims])

    def make_src_mask_enc(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_src_mask_bert(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx)

        # src_mask = [batch size, src len]

        return src_mask.float()

    def forward(self, src, atts=False):
        # src = [batch size, src len]

        src_mask = self.make_src_mask(src)

        outs = self.encoder(src, src_mask, output_attentions=atts)
        enc_src = outs.last_hidden_state
        if 'attentions' in outs:
            attentions = outs.attentions
        else:
            attentions = None

        # enc_src = [batch size, src len, hid dim]
        # attentions = [batch size, n heads, src len, src len]

        # only consider first token's representation
        enc_src = enc_src[:, 0, :]
        # enc_src = [batch size, hid dim]

        enc_src = self.fc(enc_src)
        # enc_src = [batch size, fc_dim]
        enc_src = F.relu(enc_src)
        # enc_src = [batch size, fc_dim]

        output = {f'q{idx + 1}': clf(enc_src) for idx, clf in enumerate(self.clfs)}
        # output = {q1: [batch size, out dim], ...}
        return {
            'logits': output,
            'attentions': attentions,
        }
