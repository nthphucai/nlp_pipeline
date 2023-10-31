import itertools
import random
import re
import string
from typing import Generator, List, Union


MCA_ENG_PATTERN = " and | or |, "


MCA_ENG_PATTERN = " and | or |, "

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def normalize_answer(examples: list):
    # exception case
    examples = [re.sub("hòa", "hoà", example) for example in examples]
    examples = [
        re.sub("Oa - sinh - tơn", "Oa-sinh-tơn", example) for example in examples
    ]
    examples = [re.sub("Oa - xin Tơn", "Oa-sinh-tơn", example) for example in examples]
    examples = [
        re.sub("Oa - sinh - tôn", "Oa-sinh-tơn", example) for example in examples
    ]

    return examples


def remove_hl_character(mc_answer: List[str]) -> List[str]:
    results = []
    for item in mc_answer:
        if "<hl>" in item:
            results.extend(item.split("<hl>"))
        else:
            results.append(item)
    results = [item.strip() for item in results]
    return results


def remove_duplicate_answer(examples: List[str], t: float = 0.8) -> List[str]:
    correct_answer = examples[0]
    idc = 0
    while idc < len(examples):
        key = examples[idc]
        query = [query for query in examples if query != key]
        for q in query:
            try:
                if (len(key) < len(q)) and (key in q):
                    idc = examples.index(key)
                    examples.pop(idc)
                elif (len(q) < len(key)) and (q in key):
                    idc = examples.index(q)
                    examples.pop(idc)
            except ValueError:
                continue
        idc += 1

    examples = normalize_answer(examples)
    examples = list(filter(lambda answer: answer != correct_answer, examples))
    examples.insert(0, correct_answer)
    return examples


def remove_short_answer(answers: List[str]):
    correct_answer = answers[0]
    distractors = answers[1:]
    length = len(correct_answer) // 2
    answers = list(filter(lambda answer: len(answer) > length, distractors))
    answers.insert(0, correct_answer)
    return answers


def numerical_pattern(num_pattern, correct_answer, n_distractor=2):
    num = num_pattern[0]
    distractors = []
    for _ in range(n_distractor):
        plus = random.randint(1, 9)
        distractor = re.sub(num, str(int(num) + plus), correct_answer)
        distractors.append(distractor)
    return distractors


def strip_space(mc_answers: Union[List[str], Generator]):
    distractors = mc_answers[1:]
    correct_answer = mc_answers[0]
    distractors = filter(lambda distractor: distractor != "", distractors)
    distractors = list(set([distractor.strip() for distractor in distractors]))
    distractors.insert(0, correct_answer)
    return distractors


def replace_extra_ids(mc_answers: Union[List, Generator], verbose=True):
    """
    Description: Postprocess generated distractors
    """
    if verbose:
        mc_answers = [replace_all_extra_id(item) for item in mc_answers]
        post_process_answers = list(
            itertools.chain(*[item.split("<sep>") for item in mc_answers])
        )
        post_process_answers = [
            replace_all_extra_(item) for item in post_process_answers
        ]
        post_process_answers = list(
            set([answer.strip() for answer in post_process_answers if answer != ""])
        )
        return post_process_answers
    else:
        return mc_answers


def error_correct_index(text: str, substring: str, start_index: int = 0):
    """
    Description: Locate error_correct_index
    """
    try:
        index = text.index(substring, start_index)
    except ValueError:
        index = -1
    return index


def replace_all_extra_id(text: str):
    """
    Description: replace all_extra_id in generation stage
    """
    new_text = text
    start_index_of_extra_id = 0

    while error_correct_index(new_text, "<extra_id_") >= 0:
        start_index_of_extra_id = error_correct_index(
            new_text, "<extra_id_", start_index_of_extra_id
        )
        end_index_of_extra_id = error_correct_index(
            new_text, ">", start_index_of_extra_id
        )
        new_text = (
            new_text[:start_index_of_extra_id]
            + "<sep>"
            + new_text[end_index_of_extra_id + 1 :]
        )
    return new_text


def replace_all_extra_(new_text: str):
    start_index_of_extra_id = error_correct_index(new_text, ".")
    new_text = new_text[:start_index_of_extra_id]
    return new_text


def calculate_matching_rate(question_seg: str, context_seg: str) -> float:
    question_seg = question_seg.lower().split(" ")
    context_seg = context_seg.lower().split(" ")
    count = 0
    for item in question_seg:
        if item in context_seg:
            count += 1
    return count / len(question_seg)


def rank_answer_withmatching(mc_answer: List[str]):
    correct_answer = mc_answer[0]
    order = [
        calculate_matching_rate(correct_answer, mc_answer[idc])
        for idc in range(len(mc_answer))
    ]
    index = [
        value[0]
        for value in sorted(enumerate(order), key=lambda item: item[1], reverse=True)
    ]
    return [mc_answer[idc] for idc in index]


def remove_duplicate_space(text: str) -> str:
    """
    Remove duplicate space characters in a string
    Args:
        text: (`str`) - String need to be cleaned

    Returns: (`str`) - String after cleaned

    """
    return re.sub(r"\s+", " ", text)


def remove_punc(text: str) -> str:
    """
    Remove punctuations in a string
    Args:
        text: (`str`) - text need to be removed punctuation

    Returns: (`str`) - text after remove punctuation

    """
    puncs = string.punctuation.replace("<", "").replace(">", "").replace("/", "")
    text = text.translate(str.maketrans(" ", " ", puncs))
    text = remove_duplicate_space(text)
    return text


def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        import nltk

        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return " ".join(filtered_sentence)


def normalize_distractors(distractors):
    distractors = [
        remove_stopwords(remove_punc(distractor).lower()) for distractor in distractors
    ]
    return distractors


def remove_duplicate_answers_english(
    examples: List[str],
) -> List[str]:
    examples = list(dict.fromkeys(examples))
    normalized_examples = normalize_distractors(examples)
    unique_examples = list(dict.fromkeys(normalized_examples))
    checked_examples = []
    for i, option in enumerate(normalized_examples):
        if option in unique_examples:
            checked_examples.append(examples[i])
            unique_examples.remove(option)
    return checked_examples


def remove_noise(answers):
    noises = ["Versiune", "versiune"]
    for noise in noises:
        try:
            answers.remove(noise)
        except ValueError:
            pass
    return answers


def remove_dots_answers(
    examples: List[str],
) -> List[str]:
    i = 0
    while i < len(examples):
        if "..." in examples[i]:
            examples.pop(i)
        else:
            i += 1
    return examples


def add_empty_answers(
    examples: List[str],
) -> List[str]:
    while len(examples) < 4:
        examples.append(" ")
    return examples


def _any_digit_in_string(text):
    return re.search("\d", text)


def _collect_qualified_mca_sample(qa_pair: List[dict]) -> List[dict]:
    """Using MCA_ENG_PATTERN and strict rules to choose qualified answers to create multiple-correct-answer."""
    r_pattern = MCA_ENG_PATTERN
    list_qualified_answer = []
    key_words = [" and ", " or "]
    banned_words = []
    for sample in qa_pair:
        if any(key_word in sample["answers"][0] for key_word in key_words) and not any(
            banned_word in sample["answers"][0] for banned_word in banned_words
        ):
            correct_items = re.split(r_pattern, sample["answers"][0])
            if any(
                True for item in correct_items if len(item.split(" ")) >= 3
            ) or _any_digit_in_string(sample["answers"][0]):
                continue
            else:
                list_qualified_answer.append(sample)
    return list_qualified_answer


def create_multiple_correct_qa(qa_pair: List[dict]) -> List[dict]:
    """
    Create multiple correct question-answers.
    Args:
        qa_pair (List[dict]): List of question-answer pair.

    Returns:
        List[dict]: A list of multiple-correct-answer and question pairs.
        Each pair will have 'mca' attribute to distinguish from other multiple-choice qa pairs.
        The 'answer' attribute will contain two list, the first one contains all possible correct options
        and the second one contains three incorrect options.
    """
    qualified_mca = _collect_qualified_mca_sample(qa_pair)
    multiple_correct_qa = []
    for question_answer in qualified_mca:
        question_answer = _create_multiple_correct_question(question_answer)
        question_answer = _create_multiple_correct_answers(question_answer)
        multiple_correct_qa.append(question_answer)
    return multiple_correct_qa


def _create_multiple_correct_question(question_answer: dict) -> dict:
    """Create a multiple correct question."""
    question_answer["question"] = (
        "Choose ALL correct statements: " + question_answer["question"][0]
    )
    return question_answer


def _create_multiple_correct_answers(question_answer: dict) -> dict:
    """Using MCA_ENG_PATTERN to extract multiple-correct-answer from qualified answers
    and add 'mca' attribute to the question-answer."""
    r_pattern = MCA_ENG_PATTERN
    correct_answers = re.split(r_pattern, question_answer["answers"][0])
    wrong_answers = question_answer["answers"][1:]
    question_answer["answers"] = [correct_answers, wrong_answers]
    question_answer["mca"] = True
    return question_answer
