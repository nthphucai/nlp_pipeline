from typing import List

from transformers import pipeline

from questgen.pipelines.modules.complement_distractors.distractor_base import (
    DistractorBase,
)


class DistractorComplement(DistractorBase):
    def __init__(
        self,
        fillmask_model_path: str,
        task: str = "fill-mask",
        num_grams: int = 2,
        mask_token: str = "<mask>",
        keyword_extract_strategy: str = "pos_tagging",
    ):
        super().__init__(keyword_extract_strategy, num_grams)

        self.task = task
        self.mask_token = mask_token
        self.fillmask_pipeline = pipeline(
            task=task, model=fillmask_model_path, tokenizer="vinai/phobert-base"
        )

    def _predict_mask_word(self, masked_sentence: str) -> List[dict]:
        prediction = self.fillmask_pipeline(masked_sentence)
        return prediction

    def complement(self, sentence: str, num_masks: int = 2, top_result: int = None):
        """Complement distractors in case not enough distrators generated

        Args:
            sentence (str): correct answer
            num_masks (int): number of masked distractors. Defaults to 2.
            top_result: (int): number of results have been returned according to score.
        Returns:
            str: distractors
        """
        masked_sentence = self.masking(sentence, num_masks)
        if self.task == "fill-mask":
            masked_sentence = "<s> " + masked_sentence + " </s>"

        prediction = self._predict_mask_word(masked_sentence)
        results = self.unmasking_multiple_results(
            masked_sent=masked_sentence, pred=prediction
        )
        results = sorted(results, key=lambda item: item["score"], reverse=True)
        if top_result is not None:
            results = [result["predict_sentence"] for result in results[:top_result]]
        else:
            results = [result["predict_sentence"] for result in results]
        return results
