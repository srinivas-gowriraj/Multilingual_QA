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
train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]
val_dataset = [y for dataset in [fr_train_dataset, vi_val_dataset] for y in dataset]

#Taking first 37 samples.
train1 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-HealthCareServices") or i["positive"].split("//")[-1]!= (" vi-HealthCareServices"))][:37]

train2 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-SocialSecurity") or i["positive"].split("//")[-1]!= (" vi-SocialSecurity"))][:37]
train3 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-Insurance") or i["positive"].split("//")[-1]!= (" vi-Insurance"))][:37]
train4 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-DepartmentOfMotorVehicles") or i["positive"].split("//")[-1]!= (" vi-DepartmentOfMotorVehicles"))][:37]
train5 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-Technology") or i["positive"].split("//")[-1]!= (" vi-Technology"))][:37]
train6 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-PublicServices") or i["positive"].split("//")[-1]!= (" vi-PublicServices"))][:37]
train7 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-Health") or i["positive"].split("//")[-1]!= (" vi-Health"))][:37]
train8 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-VeteranAffairs") or i["positive"].split("//")[-1]!= (" vi-VeteranAffairs"))][:37]
train9 = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-StudentFinancialAidinUSA") or i["positive"].split("//")[-1]!= (" vi-StudentFinancialAidinUSA"))][:37]

train_dataset = [x for dataset in [train1, train2, train3, train4, train5, train6, train7, train8, train9] for x in dataset]
print(len(train_dataset))
#train_dataset = [i for i in train_dataset if (i["positive"].split("//")[-1]!= (" fr-Health") or i["positive"].split("//")[-1]!= (" vi-Health"))]
val1 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-HealthCareServices") or i["positive"].split("//")[-1]!= (" vi-HealthCareServices"))][:13]
val2 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-SocialSecurity") or i["positive"].split("//")[-1]!= (" vi-SocialSecurity"))][:13]
val3 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-Insurance") or i["positive"].split("//")[-1]!= (" vi-Insurance"))][:13]
val4 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-DepartmentOfMotorVehicles") or i["positive"].split("//")[-1]!= (" vi-DepartmentOfMotorVehicles"))][:13]
val5 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-Technology") or i["positive"].split("//")[-1]!= (" vi-Technology"))][:13]
val6 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-PublicServices") or i["positive"].split("//")[-1]!= (" vi-PublicServices"))][:13]
val7 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-Health") or i["positive"].split("//")[-1]!= (" vi-Health"))][:13]
val8 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-VeteranAffairs") or i["positive"].split("//")[-1]!= (" vi-VeteranAffairs"))][:13]
val9 = [i for i in val_dataset if (i["positive"].split("//")[-1]!= (" fr-StudentFinancialAidinUSA") or i["positive"].split("//")[-1]!= (" vi-StudentFinancialAidinUSA"))][:13]
val_dataset = [x for dataset in [val1, val2, val3, val4, val5, val6, val7, val8, val9] for x in dataset]
  
all_passages = []
for file_name in ['fr']:
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
    accumulation_steps=8,
    batch_size=16,
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
