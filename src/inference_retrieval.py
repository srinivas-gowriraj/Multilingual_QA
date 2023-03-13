import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
from datasets import load_dataset

'''with open('data/dev.json') as f_in:
    with open('data/input.jsonl', 'w') as f_out:
        for line in f_in.readlines():
            sample = json.loads(line)
            sample['positive'] = ''
            sample['negative'] = ''
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

with open('data/input.jsonl') as f:
    eval_dataset = [json.loads(line) for line in f.readlines()]'''

fr_val_dataset = load_dataset('json', data_files='./data/splits/FrDoc2BotRetrieval_val.json')['train']
vi_val_dataset = load_dataset('json', data_files='./data/splits/ViDoc2BotRetrieval_val.json')['train']


eval_dataset = [x for dataset in [fr_val_dataset, vi_val_dataset] for x in dataset]

all_passages = []
for file_name in ['fr', 'vi']:
    with open(f'all_passages/{file_name}.json') as f:
        all_passages += json.load(f)

cache_path = './DAMO_ConvAI/nlp_convai_retrieval_pretrain'
trainer = DocumentGroundedDialogRetrievalTrainer(
    model=cache_path,
    train_dataset=None,
    eval_dataset=eval_dataset,
    all_passages=all_passages
)

trainer.evaluate(
    checkpoint_path=os.path.join(trainer.model.model_dir,
                                 'finetuned_model.bin'))
