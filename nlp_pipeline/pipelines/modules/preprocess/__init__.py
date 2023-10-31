from .aqg_preprocess import remove_viquad_noise
from .standard_preprocess import (
    extract_keywords,
    remove_hle,
    remove_stopwords,
    sentence_segment,
    unicode_normalize,
    word_segment,
)


__mapping__ = {
    "hle": remove_hle,
    "unicode_normalize": unicode_normalize,
    "word_segment": word_segment,
    "sent_segment": sentence_segment,
    "stop_words": remove_stopwords,
    "viquad_noise": remove_viquad_noise,
    "extract_keyword": extract_keywords,
}
