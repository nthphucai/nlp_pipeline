from .mc_postprocess import (
    add_empty_answers,
    numerical_pattern,
    rank_answer_withmatching,
    remove_dots_answers,
    remove_duplicate_answer,
    remove_duplicate_answers_english,
    remove_hl_character,
    remove_noise,
    remove_short_answer,
    replace_extra_ids,
    strip_space,
)


__mapping__ = {
    "replace_extra_id": replace_extra_ids,
    "remove_duplicate_answer": remove_duplicate_answer,
    "strip_space": strip_space,
    "remove_short_answer": remove_short_answer,
    "numerical_pattern": numerical_pattern,
    "rank_answer_withmatching": rank_answer_withmatching,
    "remove_hl_character": remove_hl_character,
    "remove_duplicate_answers_english": remove_duplicate_answers_english,
    "remove_dots_answers": remove_dots_answers,
    "add_empty_answer": add_empty_answers,
    "remove_noise": remove_noise,
}
