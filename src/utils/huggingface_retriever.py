import os

import faiss
import json
import numpy as np
import torch
from torch import nn
from typing import Dict, Any
import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer

from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import ModeKeys
from modelscope.utils.logger import get_logger

from modelscope.preprocessors import Preprocessor
from modelscope.models.base import Tensor
from modelscope.utils.constant import ModeKeys
from modelscope.utils.type_assert import type_assert

logger = get_logger()

class Wrapper(nn.Module):

    def __init__(self, encoder):
        super(Wrapper, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        return self.encoder(input_ids, attention_mask).pooler_output

class HFDPRModel(nn.Module):

    def __init__(self, checkpoint="roberta-base"):
        super().__init__()
        self.qry_encoder = Wrapper(AutoModel.from_pretrained(checkpoint))
        self.ctx_encoder = Wrapper(AutoModel.from_pretrained(checkpoint))
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
    
    def encode_query(self, input: Dict[str, Tensor]):
        query_input_ids = input['query_input_ids']
        query_attention_mask = input['query_attention_mask']
        query_vector = self.qry_encoder(query_input_ids,
                                              query_attention_mask, None)
        return query_vector

    def encode_context(self, input: Dict[str, Tensor]):
        context_input_ids = input['context_input_ids']
        context_attention_mask = input['context_attention_mask']
        context_vector = self.ctx_encoder(context_input_ids,
                                                context_attention_mask, None)
        return context_vector

def collate(batch):
    query = [item['query'] for item in batch]
    positive = [item['positive'] for item in batch]
    negative = [item['negative'] for item in batch]
    return query, positive, negative


def prepare_optimizer(model, lr, weight_decay, eps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        weight_decay,
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


def measure_result(result_dict):
    recall_k = [1, 5, 10, 20]
    meters = {f'R@{k}': [] for k in recall_k}

    for output, target in zip(result_dict['outputs'], result_dict['targets']):
        for k in recall_k:
            if target in output[:k]:
                meters[f'R@{k}'].append(1)
            else:
                meters[f'R@{k}'].append(0)
    for k, v in meters.items():
        meters[k] = sum(v) / len(v)
    return meters

# @PREPROCESSORS.register_module(
#     Fields.nlp, module_name=Preprocessors.document_grounded_dialog_retrieval_labse)
class DocumentGroundedDialogRetrievalPreprocessorHF(Preprocessor):

    def __init__(self, model_dir: str="", hf_chekcpoint = "roberta-base", *args, **kwargs):
        """The preprocessor for DGDS retrieval task, based on transformers' tokenizer.
        Args:
            model_dir: The model dir containing the essential files to build the tokenizer.
        """
        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.hf_chekcpoint = hf_chekcpoint
        # self.config = Config.from_file(
        #     os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        self.device = 'cuda' \
            if ('device' not in kwargs or kwargs['device'] == 'gpu') and torch.cuda.is_available() \
            else 'cpu'
        # self.query_sequence_length = self.config['query_sequence_length']
        # self.context_sequence_length = self.config['context_sequence_length']
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_chekcpoint)

    @type_assert(object, Dict)
    def __call__(self,
                 data: Dict[str, Any],
                 invoke_mode=ModeKeys.INFERENCE,
                 input_type='query',
                 **preprocessor_param) -> Dict[str, Any]:
        if invoke_mode in (ModeKeys.TRAIN, ModeKeys.EVAL
                           ) and invoke_mode != ModeKeys.INFERENCE:
            query, positive, negative = data['query'], data['positive'], data[
                'negative']

            query_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                query,
                padding=True,
                return_tensors='pt',
                # max_length=self.query_sequence_length,
                truncation=True)

            context_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                positive + negative,
                padding=True,
                return_tensors='pt',
                # max_length=self.context_sequence_length,
                truncation=True)

            result = {
                'query_input_ids': query_tokenizer_outputs.input_ids,
                'query_attention_mask': query_tokenizer_outputs.attention_mask,
                'context_input_ids': context_tokenizer_outputs.input_ids,
                'context_attention_mask':
                context_tokenizer_outputs.attention_mask,
                'labels':
                torch.tensor(list(range(len(query))), dtype=torch.long)
            }
        elif input_type == 'query':
            query = data['query']
            query_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                query,
                padding=True,
                return_tensors='pt',
                # max_length=self.query_sequence_length,
                truncation=True)
            result = {
                'query_input_ids': query_tokenizer_outputs.input_ids,
                'query_attention_mask': query_tokenizer_outputs.attention_mask,
            }
        else:
            context = data['context']
            context_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                context,
                padding=True,
                return_tensors='pt',
                # max_length=self.context_sequence_length,
                truncation=True)
            result = {
                'context_input_ids': context_tokenizer_outputs.input_ids,
                'context_attention_mask':
                context_tokenizer_outputs.attention_mask,
            }

        for k, v in result.items():
            result[k] = v.to(self.device)

        return result


# @TRAINERS.register_module(
#     module_name=Trainers.document_grounded_dialog_retrieval_trainer_labse)
class DocumentGroundedDialogRetrievalTrainerHF(EpochBasedTrainer):

    def __init__(self, model_save_path, device=torch.device('cuda'), hf_checkpoint="roberta-base", *args, **kwargs):
        self.hf_checkpoint = hf_checkpoint
        self.preprocessor = DocumentGroundedDialogRetrievalPreprocessorHF(hf_checkpoint=self.hf_checkpoint)
        self.device = self.preprocessor.device
        self.model = HFDPRModel(self.hf_checkpoint).to(self.device)
        self.model_save_path = model_save_path
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok = True)
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset = kwargs['eval_dataset']
        self.all_passages = kwargs['all_passages']
    
    def train(self,
              total_epoches=20,
              batch_size=128,
              per_gpu_batch_size=32,
              accumulation_steps=1,
              learning_rate=2e-5,
              warmup_ratio=0.1,
              weight_decay=0.1,
              eps=1e-06,
              loss_log_freq=40):
        """
        Fine-tuning trainsets
        """
        # obtain train loader
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate)

        optimizer = prepare_optimizer(self.model, learning_rate,
                                      weight_decay, eps)
        steps_per_epoch = len(train_loader) // accumulation_steps
        scheduler = prepare_scheduler(optimizer, total_epoches,
                                      steps_per_epoch, warmup_ratio)

        best_score = 0.0
        for epoch in range(total_epoches):
            self.model.train()
            losses = []
            for index, payload in enumerate(tqdm.tqdm(train_loader)):
                query, positive, negative = payload
                input = self.preprocessor(
                    {
                        'query': query,
                        'positive': positive,
                        'negative': negative
                    },
                    invoke_mode=ModeKeys.TRAIN)
                query_input_ids = input['query_input_ids']
                query_attention_mask = input['query_attention_mask']
                context_input_ids = input['context_input_ids']
                context_attention_mask = input['context_attention_mask']
                labels = input['labels']
                loss, logits = self.model.forward(query_input_ids,
                query_attention_mask,
                context_input_ids,
                context_attention_mask,
                labels)

                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

                loss.backward()

                if (index + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                losses.append(loss.item())
                if (index + 1) % loss_log_freq == 0:
                    logger.info(
                        f'epoch: {epoch} \t batch: {batch_size * index} \t loss: {sum(losses) / len(losses)}'
                    )
                    losses = []
            if losses:
                logger.info(
                    f'epoch: {epoch} \t batch: last \t loss: {sum(losses) / len(losses)}'
                )

            meters = self.evaluate(per_gpu_batch_size=per_gpu_batch_size)
            total_score = sum([x for x in meters.values()])
            if total_score >= best_score:
                best_score = total_score
                state_dict = self.model.state_dict()
                torch.save(state_dict, self.model_save_path)
                logger.info(
                    'epoch %d obtain max score: %.4f, saving model to %s' %
                    (epoch, total_score, self.model_save_path))

    def evaluate(self, per_gpu_batch_size=32, checkpoint_path=None):
        """
        Evaluate testsets
        """
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(state_dict)

        valid_loader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=per_gpu_batch_size,
            collate_fn=collate)
        self.model.eval()
        with torch.no_grad():
            all_ctx_vector = []
            for mini_batch in tqdm.tqdm(
                    range(0, len(self.all_passages), per_gpu_batch_size)):
                context = self.all_passages[mini_batch:mini_batch
                                            + per_gpu_batch_size]
                processed = \
                    self.preprocessor({'context': context},
                                      invoke_mode=ModeKeys.INFERENCE,
                                      input_type='context')
                sub_ctx_vector = self.model.encode_context(
                    processed).detach().cpu().numpy()
                all_ctx_vector.append(sub_ctx_vector)

            all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
            all_ctx_vector = np.array(all_ctx_vector).astype('float32')
            faiss_index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
            faiss_index.add(all_ctx_vector)

            results = {'outputs': [], 'targets': []}
            for index, payload in enumerate(tqdm.tqdm(valid_loader)):
                query, positive, negative = payload
                processed = self.preprocessor({'query': query},
                                              invoke_mode=ModeKeys.INFERENCE)
                query_vector = self.model.encode_query(
                    processed).detach().cpu().numpy().astype('float32')
                D, Index = faiss_index.search(query_vector, 20)
                results['outputs'] += [[
                    self.all_passages[x] for x in retrieved_ids
                ] for retrieved_ids in Index.tolist()]
                results['targets'] += positive
            meters = measure_result(results)
            result_path = os.path.join(os.path.dirname(self.model_save_path),
                                       'evaluate_result.json')
            with open(result_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        logger.info(meters)
        return meters