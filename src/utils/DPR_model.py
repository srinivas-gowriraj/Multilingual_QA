from __future__ import absolute_import, division, print_function
import os.path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (AutoConfig, DPRConfig, DPRQuestionEncoder,
                          MT5ForConditionalGeneration, RagTokenForGeneration,
                          XLMRobertaForSequenceClassification, XLMRobertaModel,
                          XLMRobertaTokenizer)


class Wrapper(nn.Module):

    def __init__(self, encoder):
        super(Wrapper, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        return self.encoder(input_ids, attention_mask).pooler_output


class DPRModel(nn.Module):

    def __init__(self, model_dir, config):
        super().__init__()
        self.config = config

        qry_encoder = XLMRobertaModel(
            config=AutoConfig.from_pretrained(
                os.path.join(model_dir, 'qry_encoder')))
        ctx_encoder = XLMRobertaModel(
            config=AutoConfig.from_pretrained(
                os.path.join(model_dir, 'ctx_encoder')))
        self.qry_encoder = Wrapper(qry_encoder)
        self.ctx_encoder = Wrapper(ctx_encoder)
        self.loss_fct = nn.CrossEntropyLoss()

    @staticmethod
    def encode(model, input_ids, attention_mask, gck_segment=32):
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        pooled_output = []
        for mini_batch in range(0, input_ids.shape[0], gck_segment):
            mini_batch_input_ids = input_ids[mini_batch:mini_batch
                                             + gck_segment]
            mini_batch_attention_mask = attention_mask[mini_batch:mini_batch
                                                       + gck_segment]
            mini_batch_pooled_output = checkpoint(model, mini_batch_input_ids,
                                                  mini_batch_attention_mask,
                                                  dummy_tensor)
            pooled_output.append(mini_batch_pooled_output)
        return torch.cat(pooled_output, dim=0)

    def forward(self,
                query_input_ids,
                query_attention_mask,
                context_input_ids,
                context_attention_mask,
                labels,
                gck_segment=32):
        query_vector = self.encode(self.qry_encoder, query_input_ids,
                                   query_attention_mask, gck_segment)
        context_vector = self.encode(self.ctx_encoder, context_input_ids,
                                     context_attention_mask, gck_segment)
        logits = torch.matmul(query_vector, context_vector.T)
        loss = self.loss_fct(logits, labels)
        return loss, logits