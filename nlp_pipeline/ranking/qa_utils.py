from typing import List, Optional

import numpy as np

from questgen.pipelines.modules.preprocess import __mapping__ as preprocess
from questgen.utils.utils import get_progress


def prepare_hl(keys: Optional[str], stopwords=Optional[str]) -> List:
    keys = str(keys)

    if stopwords is not None and ".npy" in stopwords:
        stopwords = list(np.load(stopwords, allow_pickle=True))

    kwords_seg = preprocess["word_segment"](keys).split(" ")
    words = list(set(map(lambda w: w.replace("_", " "), kwords_seg)))
    words = [w for w in words if len(w) > 1]

    if "," in words:
        words.remove(",")
    if stopwords is not None:
        words = [word for word in words if word not in stopwords]
    return words


def highlight_words(sent: str, words: list) -> str:
    for w in words:
        pos = sent.lower().find(w.lower())
        sent = (
            sent
            if pos < 0
            else sent[0:pos]
            + "<hl>"
            + w
            + "<hl>"
            + highlight_words(sent[pos + len(w) :], [w])
        )
    return sent


def load_sklearn_model(path):
    import pickle

    model = pickle.load(open(path, "rb"))
    return model


def save_sklearn_model(path, model):
    import pickle

    with open(path, "wb") as f:
        pickle.dump(model, f)


def select_qa(probs, data):
    idc_sorted = [
        value[0]
        for value in sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
    ]

    def qa_ranking(idc):
        return (np.round(probs[idc], 3), data[idc]["question"], data[idc]["answers"])

    for cnt in range(len(idc_sorted)):
        curr_idc = idc_sorted[cnt]
        next_idc = idc_sorted[cnt - 1]

        if cnt == 0:
            result = qa_ranking(curr_idc)
        if (cnt > 0) and (data[curr_idc]["answers"] != data[next_idc]["answers"]):
            if data[curr_idc]["question"] != data[next_idc]["question"]:
                result = qa_ranking(curr_idc)

            yield result


def extract_features(dataloader, model, verbose=True, device="cuda") -> np.array:
    temp_lst = []
    model = model.encoder
    model = model.to(device)
    for _, batch in enumerate(get_progress(iterable=dataloader, disable=verbose)):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        features = model(input_ids, attention_mask)
        features = features.last_hidden_state.mean(axis=1)
        temp_lst.append(features.detach().cpu().numpy())

    return np.concatenate(temp_lst)
