import os
import json

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
from modelscope.utils.constant import DownloadMode
from datasets import load_dataset

fr_train_dataset = load_dataset('json', data_files="data/splits/FrDoc2BotRetrieval_train.json")["train"]
fr_val_dataset = load_dataset('json', data_files="data/splits/FrDoc2BotRetrieval_val.json")["train"]


vi_train_dataset = load_dataset('json', data_files="data/splits/ViDoc2BotRetrieval_train.json")["train"]
vi_val_dataset = load_dataset('json', data_files="data/splits/ViDoc2BotRetrieval_val.json")["train"]

# fr_train_dataset_origin = MsDataset.load(
#     'DAMO_ConvAI/FrDoc2BotRetrieval',
#     download_mode=DownloadMode.FORCE_REDOWNLOAD)

# vi_train_dataset_origin  = MsDataset.load(
#     'DAMO_ConvAI/ViDoc2BotRetrieval',
#     download_mode=DownloadMode.FORCE_REDOWNLOAD)

# breakpoint()

train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]
val_dataset = [y for dataset in [fr_val_dataset, vi_val_dataset] for y in dataset]

all_passages = []
for file_name in ['fr', 'vi']:
    with open(f'all_passages/{file_name}.json') as f:
        all_passages += json.load(f)

cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir='./')
trainer = DocumentGroundedDialogRetrievalTrainer(
    model=cache_path,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    all_passages=all_passages)

trainer.train(
    # batch_size=128,
    accumulation_steps=4,
    batch_size=32,
    total_epoches=50,
)
trainer.evaluate(
    checkpoint_path=os.path.join(trainer.model.model_dir,
                                 'finetuned_model.bin'))


# ret -> top 20 pids
# reranker -> cross-encoder query, top 20 (contains gold R@20=1), (q+pi) cls repr, cls - linear - score, for all 20 pairs, loss = exp(q+p_gold)/sum(exp(q+pi)) 
#   how to chose: p- -> at entire index level 
    # train better reranker using french, negatives pids are non-gold retrieved pids

# not train anything, just evaluate using their checkpoint without training
# 
