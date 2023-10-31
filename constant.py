import os

PARENT_PATH = os.getcwd()

WORKERS = 5
BOOK_PATH = "data/history_textbook_highschool_vn_w_grades_data.json"

SCRIPT_PATH = os.path.join(
    PARENT_PATH,
    "nlp_pipeline/dataset/build_transformer_format_dataset/build_transformer_format_dataset.py",
)
VI_VECTOR_PATH = os.path.join(PARENT_PATH, "nlp_pipeline/pipelines/modules/summarize/vi.vec")
STOPWORD_PATH = os.path.join(PARENT_PATH, "nlp_pipeline/dataset/create_data/history_stopwords.npy")

ENGLISH_MULTICHOICE_MODEL_URL = "http://minio.dev.ftech.ai/fschool-english-multiple-choice-questgen-v1.0-ae736037/fschool_english_multiple_questgen_v1.0.zip"
ENGLISH_MULTITASK_MODEL_URL = "http://minio.dev.ftech.ai/fschool-english-simple-questgen-v1.1-21e842d1/fschool_english_simple_questgen_v1.1.zip"

HISTORY_MULTICHOICE_MODEL_URL = "http://minio.dev.ftech.ai/fschool-history-multiple-choice-questgen-v1.3-5d97124a/fschool_history_multiple_questgen_v1.3.zip"
HISTORY_MULTITASK_MODEL_URL = "http://minio.dev.ftech.ai/fschool-history-simple-questgen-v1.4-966ffe82/fshool_history_simple_questgen_v1.4.zip"

FQA_MULTITASK_MODEL_URL = "http://minio.dev.ftech.ai/fschool-question-generation-1.1.0-b0463c36/AQG_Multitask_Transformers_v4.25.1.zip"


MULTICHOICE_MODEL = "fschool_english_multiple_questgen_v1.0"
MULTITASK_MODEL = "fschool_english_simple_questgen_v1.1"

PRETRAINED_MODEL_URL_MAP = {
    "english": {
        "multitask": ENGLISH_MULTITASK_MODEL_URL,
        "mc": ENGLISH_MULTICHOICE_MODEL_URL,
    },
    "history": {
        "multitask": HISTORY_MULTITASK_MODEL_URL,
        "mc": HISTORY_MULTICHOICE_MODEL_URL,
    },
    "fqa": {"multitask": FQA_MULTITASK_MODEL_URL, "mc": None},
}
