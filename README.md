# Multilingual_QA

0. In `src/config.py`, modify value for `data_dir` to set data directory paths according to your file system. All other data paths are relative to `data_dir`
1. `conda activate <venv_name>`
2. `cd src`
3. `python 0_download_data_splits.py`
4. Modify the arguments in `config.py` as needed
5. Make sure that all the data files are present in the base directory folder specified by `config.py`



## Training
### Retrieval
```bash
python train_retrieval.py -mt <model_type> -l <train_languages_list> -ofp <output_file_path_where_to_store_model_weights>
```
This produces the model **DAMO_ConvAI/nlp_convai_retrieval_pretrain/**


### Rerank
```bash
python train_rerank.py 
```
This produces the model **output/**


### Generation
```bash
python train_generation.py
```
This produces the model **DAMO_ConvAI/nlp_convai_generation_pretrain/**

## Testing
### Retrieval
```bash
python inference_retrieval.py -mt <model_type> -l <train_languages_list> -mc <output_file_path_where_model_weights_stored>
```
This produces the input file **input.jsonl** and the retrieval result **output_folder_path_where_model_weights_stored/evaluate_result.json**


### Rerank
```bash
python inference_rerank.py -rpfp <file_path_to_output_jsonl_from_inference_retrieval.json>
```
This produces the rerank result **rerank_output.jsonl**

### Generation
```bash
python inference_generation.py
```
This produces the generation result **outputStandardFile.json**, you can submit the document directly.