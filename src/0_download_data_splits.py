from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import os

from config import hparams
hp = hparams()

def main():
    for language in hp.lang_data_paths:
        lang_paths =  hp.lang_data_paths[language]
        for stage in lang_paths:
            if stage == "short_name":
                continue
            dataset = MsDataset.load(lang_paths[stage]["modelscope_url"], download_mode=DownloadMode.FORCE_REDOWNLOAD)
            hf_dataset  = dataset.to_hf_dataset().train_test_split(test_size = hp.test_size, seed = hp.seed)
            hf_train_dataset = hf_dataset['train']
            hf_val_dataset = hf_dataset['test']
            os.makedirs(lang_paths[stage]['path'], exist_ok=True)
            hf_train_dataset.to_json(f"{lang_paths[stage]['path']}_train.json")
            hf_val_dataset.to_json(f"{lang_paths[stage]['path']}_val.json")

if __name__ == "__main__":
    main()