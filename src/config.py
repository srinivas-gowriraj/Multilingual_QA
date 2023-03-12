import os
class hparams:
    seed=11797
    test_size = 0.2
    data_dir = "/home/sohamdit/Multilingual_QA/src/data"
    lang_data_paths = {
        "french": {
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
            "short_name": "fr"
        },
        "vietnamese": {
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
            "short_name": "vi"
        }
    }
