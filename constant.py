import os

INSTRUCTION = "Summarize the following"


PARENT_PATH = os.getcwd()

WORKERS = 5

SCRIPT_PATH = os.path.join(
    PARENT_PATH,
    "nlp_pipeline/dataset/build_transformer_format_dataset/build_transformer_format_dataset.py",
)
VI_VECTOR_PATH = os.path.join(PARENT_PATH, "nlp_pipeline/pipelines/modules/summarize/vi.vec")
STOPWORD_PATH = os.path.join(PARENT_PATH, "nlp_pipeline/dataset/create_data/history_stopwords.npy")

