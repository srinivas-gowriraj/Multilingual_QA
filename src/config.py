import os


class hparams:
    def __init__(self):
        self.seed = 11797
        self.test_size = 0.2
        self.root_dir = "/home/sohamdit/Multilingual_QA/"
        self.data_dir = os.path.join(self.root_dir, "src", "data")
        self.all_passages_dir = os.path.join(self.data_dir, "all_passages")
        self.num_bm25_docs = 10
        self.data_processing_stages = [0, 1, 2, 3, 4]
        self.leaderboard_input_file = os.path.join(self.data_dir, "dev.json")
        self.lang_data_paths = {
            "french": {
                "preprocessed_available": True,
                "stages": {
                    "retrieval": {
                        "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRetrieval',
                        "path": os.path.join(self.data_dir, "splits", "FrDoc2BotRetrieval"),
                    },
                    "rerank": {
                        "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRerank',
                        "path": os.path.join(self.data_dir, "splits", "FrDoc2BotRerank"),
                    },
                    "generation": {
                        "modelscope_url": 'DAMO_ConvAI/FrDoc2BotGeneration',
                        "path": os.path.join(self.data_dir, "splits", "FrDoc2BotGeneration"),
                    },
                },
                "short_name": "fr",
                "passage_path": os.path.join(self.data_dir, "all_passages", "fr.json")
            },
            "french_gpt": {
                "preprocessed_available": True,
                "stages": {
                    "retrieval": {
                        # "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRetrieval',
                        "path": os.path.join(self.data_dir, "gpt_splits", "FrDoc2BotRetrieval_gpt"),
                    },
                    # "rerank": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRerank',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotGeneration',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotGeneration"),
                    # },
                },
                "short_name": "fr_gpt",
                "passage_path": os.path.join(self.data_dir, "all_passages", "fr.json")
            },
            "french_2k_raw": {
                # "preprocessed_available": True,
                "stages": {
                    "retrieval": {
                        # "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRetrieval',
                        "path": os.path.join(self.data_dir, "gpt_splits", "FrDoc2BotRetrieval"),
                    },
                    # "rerank": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRerank',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotGeneration',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotGeneration"),
                    # },
                },
                "short_name": "fr_2kr",
                "passage_path": os.path.join(self.data_dir, "all_passages", "fr.json")
            },
             "french_2k_last_turn": {
                "preprocessed_available": True,
                "stages": {
                    "retrieval": {
                        # "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRetrieval',
                        "path": os.path.join(self.data_dir, "gpt_splits", "FrDoc2BotRetrieval_2k_last_turn"),
                    },
                    # "rerank": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRerank',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotGeneration',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotGeneration"),
                    # },
                },
                "short_name": "fr_2kl",
                "passage_path": os.path.join(self.data_dir, "all_passages", "fr.json")
            },
            "vietnamese": {
                "preprocessed_available": True,
                "stages": {
                    "retrieval": {
                        "modelscope_url": 'DAMO_ConvAI/ViDoc2BotRetrieval',
                        "path": os.path.join(self.data_dir, "splits", "ViDoc2BotRetrieval"),
                    },
                    "rerank": {
                        "modelscope_url": 'DAMO_ConvAI/ViDoc2BotRerank',
                        "path": os.path.join(self.data_dir, "splits", "ViDoc2BotRerank"),
                    },
                    "generation": {
                        "modelscope_url": 'DAMO_ConvAI/ViDoc2BotGeneration',
                        "path": os.path.join(self.data_dir, "splits", "ViDoc2BotGeneration"),
                    },
                },
                "short_name": "vi",
                "passage_path": os.path.join(self.data_dir, "all_passages", "vi.json")
            },
            "vietnamese_gpt": {
                "preprocessed_available": True,
                "stages": {
                    "retrieval": {
                        # "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRetrieval',
                        "path": os.path.join(self.data_dir, "gpt_splits", "ViDoc2BotRetrieval_gpt"),
                    },
                    # "rerank": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRerank',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotGeneration',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotGeneration"),
                    # },
                },
                "short_name": "vi_gpt",
                "passage_path": os.path.join(self.data_dir, "all_passages", "vi.json")
            },
            "vietnamese_2k_raw": {
                "preprocessed_available": True,
                "stages": {
                    "retrieval": {
                        # "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRetrieval',
                        "path": os.path.join(self.data_dir, "gpt_splits", "ViDoc2BotRetrieval"),
                    },
                    # "rerank": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRerank',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotGeneration',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotGeneration"),
                    # },
                },
                "short_name": "vi_2kr",
                "passage_path": os.path.join(self.data_dir, "all_passages", "vi.json")
            },
            "vietnamese_2k_last_turn": {
                "preprocessed_available": True,
                "stages": {
                    "retrieval": {
                        # "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRetrieval',
                        "path": os.path.join(self.data_dir, "gpt_splits", "ViDoc2BotRetrieval_2k_last_turn"),
                    },
                    # "rerank": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotRerank',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "modelscope_url": 'DAMO_ConvAI/FrDoc2BotGeneration',
                    #     "path": os.path.join(self.data_dir, "splits", "FrDoc2BotGeneration"),
                    # },
                },
                "short_name": "vi_2kl",
                "passage_path": os.path.join(self.data_dir, "all_passages", "vi.json")
            },
            "chinese": {
                "preprocessed_available": False,
                "modelscope_url": 'DAMO_ConvAI/ZhDoc2BotDialogue',
                "default_data_path": os.path.join(self.data_dir, "splits", "ZhDoc2BotDialogue"),
                "stages": {
                    "retrieval": {
                        "path": os.path.join(self.data_dir, "splits", "ZhDoc2BotRetrieval"),
                    },
                    "rerank": {
                        "path": os.path.join(self.data_dir, "splits", "ZhDoc2BotRerank"),
                    },
                    "generation": {
                        "path": os.path.join(self.data_dir, "splits", "ZhDoc2BotGeneration"),
                    },
                },
                "short_name": "zh",
                "passage_path": os.path.join(self.data_dir, "all_passages", "zh.json")
            },
            "english": {
                "preprocessed_available": False,
                "modelscope_url": 'DAMO_ConvAI/EnDoc2BotDialogue',
                "default_data_path": os.path.join(self.data_dir, "splits", "EnDoc2BotDialogue"),
                "stages": {
                    "retrieval": {
                        "path": os.path.join(self.data_dir, "splits", "EnDoc2BotRetrieval"),
                    },
                    "rerank": {
                        "path": os.path.join(self.data_dir, "splits", "EnDoc2BotRerank"),
                    },
                    "generation": {
                        "path": os.path.join(self.data_dir, "splits", "EnDoc2BotGeneration"),
                    },
                },
                "short_name": "en",
                "passage_path": os.path.join(self.data_dir, "all_passages", "en.json")
            },
            "english_gpt": {
                "preprocessed_available": False,
                # "modelscope_url": 'DAMO_ConvAI/EnDoc2BotDialogue',
                # "default_data_path": os.path.join(self.data_dir, "splits", "EnDoc2BotDialogue"),
                "stages": {
                    "retrieval": {
                        "path": os.path.join(self.data_dir, "gpt_splits", "EnDoc2BotRetrieval_gpt"),
                    },
                    # "rerank": {
                    #     "path": os.path.join(self.data_dir, "splits", "EnDoc2BotRerank_gpt"),
                    # },
                    # "generation": {
                    #     "path": os.path.join(self.data_dir, "splits", "EnDoc2BotGeneration_gpt"),
                    # },
                },
                "short_name": "en_gpt",
                "passage_path": os.path.join(self.data_dir, "all_passages", "en.json")
            },
            "english_2k_last_turn": {
                "preprocessed_available": False,
                # "modelscope_url": 'DAMO_ConvAI/EnDoc2BotDialogue',
                # "default_data_path": os.path.join(self.data_dir, "splits", "EnDoc2BotDialogue"),
                "stages": {
                    "retrieval": {
                        "path": os.path.join(self.data_dir, "gpt_splits", "EnDoc2BotRetrieval_2k_last_turn"),
                    },
                    # "rerank": {
                    #     "path": os.path.join(self.data_dir, "splits", "EnDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "path": os.path.join(self.data_dir, "splits", "EnDoc2BotGeneration"),
                    # },
                },
                "short_name": "en_2kl",
                "passage_path": os.path.join(self.data_dir, "all_passages", "en.json")
            },
            "english_conqrr_rewritten_true": {
                "preprocessed_available": False,
                # "modelscope_url": 'DAMO_ConvAI/EnDoc2BotDialogue',
                # "default_data_path": os.path.join(self.data_dir, "splits", "EnDoc2BotDialogue"),
                "stages": {
                    "retrieval": {
                        "path": os.path.join(self.data_dir, "gpt_splits", "EnDoc2BotRetrieval_conqrr_rewritten_true"),
                    },
                    # "rerank": {
                    #     "path": os.path.join(self.data_dir, "splits", "EnDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "path": os.path.join(self.data_dir, "splits", "EnDoc2BotGeneration"),
                    # },
                },
                "short_name": "en_crt",
                "passage_path": os.path.join(self.data_dir, "all_passages", "en.json")
            },
             "english_conqrr_rewritten_false": {
                "preprocessed_available": False,
                # "modelscope_url": 'DAMO_ConvAI/EnDoc2BotDialogue',
                # "default_data_path": os.path.join(self.data_dir, "splits", "EnDoc2BotDialogue"),
                "stages": {
                    "retrieval": {
                        "path": os.path.join(self.data_dir, "gpt_splits", "EnDoc2BotRetrieval_conqrr_rewritten_false"),
                    },
                    # "rerank": {
                    #     "path": os.path.join(self.data_dir, "splits", "EnDoc2BotRerank"),
                    # },
                    # "generation": {
                    #     "path": os.path.join(self.data_dir, "splits", "EnDoc2BotGeneration"),
                    # },
                },
                "short_name": "en_crf",
                "passage_path": os.path.join(self.data_dir, "all_passages", "en.json")
            }
        }
        self.available_languages = list(self.lang_data_paths.keys())
        self.available_languages_short_names = [
            self.lang_data_paths[lang]["short_name"] for lang in self.available_languages]
