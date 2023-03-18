import os
class hparams:
    seed=11797
    test_size = 0.2
    root_dir = "/home/sohamdit/Multilingual_QA/"
    data_dir = os.path.join(root_dir, "src", "data")
    all_passages_dir = os.path.join(root_dir, "src", "all_passages")
    num_bm25_docs = 10
    available_languages = ["french", "vietnamese", "chinese", "english"]
    lang_data_paths = {
        "french": {
            "preprocessed_available": True,
            "retrieval":{ 
                "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRetrieval',
                "path": os.path.join(data_dir, "splits", "FrDoc2BotRetrieval"),
            },
            "rerank":{ 
                "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRerank',
                "path": os.path.join(data_dir, "splits", "FrDoc2BotRerank"),
            },
            "generation":{ 
                "modelscope_url": 'DAMO_ConvAI/FrDoc2BotGeneration',
                "path": os.path.join(data_dir, "splits", "FrDoc2BotGeneration"),
            },
            "short_name": "fr",
            "passage_path": os.path.join(data_dir, "all_passages", "fr.json")
        },
        "vietnamese": {
            "preprocessed_available": True,
            "retrieval":{ 
                "modelscope_url": 'DAMO_ConvAI/ViDoc2BotRetrieval',
                "path": os.path.join(data_dir, "splits", "ViDoc2BotRetrieval"),
            },
            "rerank":{ 
                "modelscope_url": 'DAMO_ConvAI/ViDoc2BotRerank',
                "path": os.path.join(data_dir, "splits", "ViDoc2BotRerank"),
            },
            "generation":{ 
                "modelscope_url": 'DAMO_ConvAI/ViDoc2BotGeneration',
                "path": os.path.join(data_dir, "splits", "ViDoc2BotGeneration"),
            },
            "short_name": "vi",
            "passage_path": os.path.join(data_dir, "all_passages", "vi.json")
        },
        "chinese": {
            "preprocessed_available": False,
            "modelscope_url": 'DAMO_ConvAI/ZhDoc2BotDialogue',
            "default_data_path": os.path.join(data_dir, "splits", "ZhDoc2BotDialogue"),
            "retrieval":{ 
                "path": os.path.join(data_dir, "splits", "ZhDoc2BotRetrieval"),
            },
            "rerank":{ 
                "path": os.path.join(data_dir, "splits", "ZhDoc2BotRerank"),
            },
            "generation":{ 
                "path": os.path.join(data_dir, "splits", "ZhDoc2BotGeneration"),
            },
            "short_name": "zh",
            "passage_path": os.path.join(data_dir, "all_passages", "zh.json")
        },
        "english": {
            "preprocessed_available": False,
            "modelscope_url": 'DAMO_ConvAI/EnDoc2BotDialogue',
            "default_data_path": os.path.join(data_dir, "splits", "EnDoc2BotDialogue"),
            "retrieval":{ 
                "path": os.path.join(data_dir, "splits", "EnDoc2BotRetrieval"),
            },
            "rerank":{ 
                "path": os.path.join(data_dir, "splits", "EnDoc2BotRerank"),
            },
            "generation":{ 
                "path": os.path.join(data_dir, "splits", "EnDoc2BotGeneration"),
            },
            "short_name": "en",
            "passage_path": os.path.join(data_dir, "all_passages", "en.json")
        }
    }
