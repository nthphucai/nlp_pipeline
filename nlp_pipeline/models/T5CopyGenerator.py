# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" T5 model with copy mechanism """
import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Model
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class T5CopyGenerator(T5ForConditionalGeneration):
    """
    MBart with the copy mechanism of (See, 2017).
    Background section: https://aclanthology.org/2020.acl-main.125.pdf
    """

    def __init__(self, config):
        super().__init__(config)

        self.model = T5Model(config)

        # Layers to compute the cross-attention
        self.attn_layer = nn.Linear(self.config.d_model, 1, bias=True)

        # Layers to compute p_gen
        self.pgen_context_layer = nn.Linear(self.config.d_model, 1, bias=True)
        self.pgen_decoder_output_layer = nn.Linear(self.config.d_model, 1, bias=True)
        self.pgen_decoder_prev_output_layer = nn.Linear(
            self.config.d_model, 1, bias=True
        )

        # Initialize weights and apply final processing
        self.init_weights()

    def _compute_cross_attn_prob(self, e, encoder_attentions=None):
        """
        Given e from Eq. 3, compute \alpha from Eq. 4.
        This method can be overwritten to include additional
        information before computing the softmax, e.g. TF-IDF or centrality.
        Args:
            e (torch.Tensor): (batch_size, target_len, source_len), the e values
               for each (target_i, source_j) for each sample in a batch.
            encoder_attentions (torch.Tensor): (batch_size, source_len, target_len),
                                               needed to compute centrality.
        Returns:
            torch.Tensor: (batch_size, target_len, source_len), the \alpha values
            of the cross-attention for each (target_i, source_j) for each sample in a batch.
        """

        # Whether to use centrality as additional information.
        if self.config.centrality:
            # Sum columns of the attentions from the last encoder layer (in-degree centrality)
            centrality_scores = encoder_attentions[-1].mean(dim=1).mean(dim=1)
            centrality_scores = centrality_scores.unsqueeze(1).repeat_interleave(
                e.size(1), dim=1
            )

            # Fix the size of the centrality scores to match the size of the e values (beam search)
            if centrality_scores.shape[0] != e.shape[0]:
                centrality_scores = centrality_scores.repeat_interleave(
                    e.shape[0] // centrality_scores.shape[0], dim=0
                )

            # Add to e the centrality scores
            e += centrality_scores

        # Whether to use tf-idf as additional information.
        if self.config.tf_idf:
            # TODO
            pass

        return nn.Softmax(dim=-1)(e)

    @staticmethod
    def _shift_right_one_pad(x):
        """
        Shift a vector one position to the right and padd.
        """
        shifted = x.roll(1)
        shifted[0] = 0
        return shifted

    def _compute_output_dist(self, encoder_outputs, decoder_outputs, encoder_input_ids):
        """
        Compute the output distribution using the copy mechanism of (See, 2017).
        Background section of: https://aclanthology.org/2020.acl-main.125.pdf
        Args:
            encoder_outputs (torch.Tensor): (batch_size, source_len, d_model)
            decoder_outputs (torch.Tensor): (batch_size, target_len, d_model)
            encoder_input_ids (torch.LongTensor): (batch_size, source_len)
        Returns:
            torch.Tensor: (batch_size, target_len, vocab_size) distribution over the vocabulary
                          computed using a copy mechanism.
        """
        encoder_attentions = encoder_outputs.attentions
        encoder_outputs = encoder_outputs[0]
        decoder_outputs = decoder_outputs[0]
        source_len = encoder_outputs.shape[1]
        target_len = decoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]

        """
        Project the encoder and decoder outputs to compute the cross-attention (Eq. 3)
        In my experiments, not to project the encoder outputs seems to work better.
        You can define `proj_enc_layer` and `proj_dec_layer` in self,
        to project the encoder outputs. If so, you will likely need to pass a `d_proj`
        argument in the config object.
        """
        proj_enc = encoder_outputs  # self.proj_enc_layer(encoder_outputs)
        proj_dec = decoder_outputs  # self.proj_dec_layer(decoder_outputs)

        # Sum the projected outputs and apply f_act to compute the cross-attention (Eq. 3)
        sum_projs = torch.nn.GELU()(
            (proj_dec[:, :, None, :] + proj_enc[:, None, :, :]).view(
                (batch_size, target_len, source_len, self.config.d_model)
            )
        )

        # Compute the cross-attentions (e and \alpha, Eqs. 3 and 4)
        e = self.attn_layer(sum_projs).squeeze(-1)
        """
        The attention to the pad token should be 0 --> e=-100 where input_ids==pad_token_id
        Tokens like stopwords can be removed in this point.
        """
        e[:, :, (encoder_input_ids == self.config.pad_token_id).nonzero()] = -100
        attns = self._compute_cross_attn_prob(e, encoder_attentions)

        # Compute the context vectors (Eq. 5)
        context_vectors = torch.einsum("ijk, ikf -> ijf", attns, encoder_outputs)

        """
        Compute P_vocab (Eq. 6)
        I used the pretrained lm_head to project both the decoder outputs
        and the context vectors.
        """
        p_vocab_decoder = self.lm_head(decoder_outputs)  # + self.final_logits_bias
        p_vocab_context = self.lm_head(context_vectors)  # + self.final_logits_bias
        p_vocab = p_vocab_decoder + p_vocab_context
        p_vocab = nn.Softmax(dim=-1)(p_vocab)

        """
        Compute p_gen (Eq. 8)
        Since there is not "state" in Transformers, I consider the
        decoder output in the current and previous steps, along with
        the context vector of the current decoder state.
        """
        pgen_context = self.pgen_context_layer(context_vectors)
        pgen_decoder_output = self.pgen_decoder_output_layer(decoder_outputs)
        pgen_decoder_prev_output = self.pgen_decoder_prev_output_layer(
            T5CopyGenerator._shift_right_one_pad(decoder_outputs)
        )
        p_gen = nn.Sigmoid()(
            pgen_context + pgen_decoder_output + pgen_decoder_prev_output
        )
        """
        In my experiments using pre-trained models, I see that `p_gen` is approximately 1 since
        the beginning of the training process. Sometimes, it worked better to fix the `p_gen`
        to the % of novel tokens.
        """
        # p_gen = torch.zeros_like(p_gen) + 0.7

        # Compute P_copy (Eq. 9)
        p_copy = torch.zeros_like(p_vocab)

        # Fix the size of the encoder_ids if beam search is being used.
        if encoder_input_ids.shape[0] != batch_size:
            encoder_input_ids = encoder_input_ids.repeat_interleave(
                batch_size // encoder_input_ids.shape[0], dim=0
            )

        p_copy = p_copy.scatter_add(
            -1,
            encoder_input_ids.repeat_interleave(attns.shape[1], dim=0).view(
                batch_size, target_len, -1
            ),
            attns,
        )

        # The output distribution is the sum of p_copy and p_vocab weighted by p_gen
        eps = torch.finfo(((1.0 - p_gen) * p_copy + p_gen * p_vocab).dtype).eps
        final_dist = torch.log((1.0 - p_gen) * p_copy + p_gen * p_vocab + eps)

        # print("P_COPY:", p_copy[0][-1].topk(20).indices)
        # print("P_VOCAB:", p_vocab[0][-1].topk(20).indices)
        # print("P_FINAL", final_dist[0][-1].topk(20).indices)

        return final_dist

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        # lm_logits = self.lm_head(sequence_output)
        lm_logits = self._compute_output_dist(
            encoder_outputs, decoder_outputs, input_ids
        )

        loss = None
        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        # if past is None:
        #     logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
        #     return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past
