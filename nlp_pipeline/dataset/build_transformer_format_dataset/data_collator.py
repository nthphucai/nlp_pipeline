from typing import Dict, List

import torch


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessary because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
class Text2TextDataCollator:
    def __init__(self, tokenizer, model_type="t5", mode="training", using_tpu=False):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mode = mode
        self.using_tpu = using_tpu

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors

        """
        batch = self._generate_batch(batch=batch, padding="longest")
        input_ids = batch["source_ids"]
        target_ids = batch["target_ids"]
        attention_mask = batch["attention_mask"]

        if len(batch.keys()) > 3:
            node_features = batch["node_features"]
            edge_index = batch["edge_index"]
            edge_relation = batch["edge_relation"]

        pad_token_id = self.tokenizer.pad_token_id

        # don't trim on tpu, for some reason trimming leads to slower training on TPU
        if not self.using_tpu:
            input_ids, attention_mask = trim_batch(
                input_ids, pad_token_id, attention_mask=attention_mask
            )
            target_ids = trim_batch(target_ids, pad_token_id)

        if self.model_type == "t5":
            lm_labels = target_ids.clone()
            decoder_input_ids = self._shift_right_t5(lm_labels)
            if self.mode == "training":
                lm_labels[lm_labels[:, :] == pad_token_id] = -100
        else:
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
            if self.mode == "training":
                lm_labels[target_ids[:, 1:] == pad_token_id] = -100

        params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids,
        }
        """
        In case of integrating graph convolution network (gcn).
        Examples:
            node_features: torch.randn(2, 8, 64) , # bz, num_entites, embedding_size
            edge_index: torch.randint(1, 8, (2, 16)) , # 2, bz * num_entites = 4 * 8
            edge_relation: torch.randn(16) # bz * num_entites = 4 * 8
        """
        if len(batch.keys()) > 3:
            params.update(
                {
                    "node_features": node_features,
                    "edge_index": edge_index,
                    "edge_relation": edge_relation,
                }
            )

        return params

    def _shift_right_t5(self, input_ids):
        decoder_start_token_id = self.tokenizer.pad_token_id
        pad_token_id = self.tokenizer.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually \
            set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `labels` has only positive values and -100"

        return shifted_input_ids

    def _generate_batch(self, batch: List, padding="longest") -> Dict:
        from torch.nn.utils.rnn import pad_sequence

        if padding == "longest":
            input_ids = pad_sequence(
                [example["source_ids"] for example in batch], batch_first=True
            )
            target_ids = pad_sequence(
                [example["target_ids"] for example in batch], batch_first=True
            )
            attention_mask = pad_sequence(
                [example["attention_mask"] for example in batch], batch_first=True
            )

            if len(batch[0].keys()) > 3:
                node_features = pad_sequence(
                    [example["source_subject"] for example in batch], batch_first=True
                )

                edge_index = pad_sequence(
                    [example["edge_index"] for example in batch], batch_first=True
                )

                edge_relation = pad_sequence(
                    [example["source_rel"] for example in batch], batch_first=True
                )

        else:
            input_ids = torch.stack([example["source_ids"] for example in batch])
            target_ids = torch.stack([example["target_ids"] for example in batch])
            attention_mask = torch.stack(
                [example["attention_mask"] for example in batch]
            )

            if len(batch[0].keys()) > 3:
                node_features = torch.stack(
                    [example["source_subject"] for example in batch]
                )
                edge_index = torch.stack([example["edge_index"] for example in batch])
                edge_relation = torch.stack(
                    [example["source_rel"] for example in batch]
                )

        encoded_batch = {
            "source_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
        }
        if len(batch[0].keys()) > 3:
            encoded_batch.update(
                {
                    "node_features": node_features,
                    "edge_index": edge_index,
                    "edge_relation": edge_relation,
                }
            )

        return encoded_batch
