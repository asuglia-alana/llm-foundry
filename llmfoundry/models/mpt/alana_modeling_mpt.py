import warnings
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from llmfoundry.models.mpt.modeling_mpt import MPTModel


class AlanaMPTModel(MPTModel):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if attention_mask is not None:
            attention_mask = attention_mask.bool()  # type: ignore

        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()  # type: ignore

        # These args are passed in by keyword in huggingface's generate function
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L2201-L2206
        # but have not yet been fully implemented in MPTModel
        if not return_dict:
            raise NotImplementedError(
                "return_dict False is not implemented yet for MPT"
            )
        if output_attentions:
            if self.attn_impl != "torch":
                raise NotImplementedError(
                    "output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`."
                )

        if (
            self.training
            and attention_mask is not None
            and attention_mask[:, 0].sum() != attention_mask.shape[0]
        ):
            raise NotImplementedError(
                "MPT does not support training with left padding."
            )

        if self.prefix_lm and prefix_mask is None:
            raise ValueError(
                "prefix_mask is a required argument when MPT is configured with prefix_lm=True."
            )

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    "sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True "
                    + "and the model is in train mode."
                )
            elif (self.attn_uses_sequence_id is False) and (sequence_id is not None):
                warnings.warn(
                    "MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. "
                    + "This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True."
                )

        S = inputs_embeds.size(1)

        assert (
            S <= self.config.max_seq_len
        ), f"Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}"

        if self.learned_pos_emb:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(
                        "past_key_values must provide a past_key_value for each attention "
                        + f"layer in the network ({len(past_key_values)=}; {self.config.n_layers=})."
                    )
                # For attn_impl: triton and flash the past key tensor spec is (batch, seq, dim).
                # For attn_impl: torch the past key tensor spec is (batch, heads, head_dim, seq).
                # Here we shift position embedding using the `seq` dim of the past key
                past_position = past_key_values[0][0].size(1)
                if self.attn_impl == "torch":
                    past_position = past_key_values[0][0].size(3)

            if S + past_position > self.config.max_seq_len:
                raise ValueError(
                    f"Cannot forward input with past sequence length {past_position} and current sequence length "
                    + f"{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}."
                )
            pos = torch.arange(
                past_position,
                S + past_position,
                dtype=torch.long,
                device=inputs_embeds.device,
            ).unsqueeze(0)
            if attention_mask is not None:
                # adjust the position indices to account for padding tokens
                pos = torch.clamp(
                    pos
                    - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[
                        :, past_position:
                    ],
                    min=0,
                )

            pos_emb = self.wpe(pos)
            x = inputs_embeds + pos_emb
        else:
            # ALiBi and NoPE use this path (RoPE will also use this path if / when enabled)
            x = inputs_embeds

        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction) + (
                x.detach() * (1 - self.embedding_fraction)
            )
            assert isinstance(self.emb_drop, nn.Module)  # pyright
            x = self.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=torch.float32,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
        )

        # initialize the past key values cache if it should be used
        presents = () if use_cache else None
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]  # type: ignore

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for b_idx, block in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = (
                past_key_values[b_idx] if past_key_values is not None else None
            )
            x, attn_weights, present = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                attention_mask=attention_mask,
                is_causal=self.is_causal,
                output_attentions=bool(output_attentions),
            )
            if presents is not None:
                presents += (present,)

            if output_attentions:
                assert all_self_attns is not None  # pyright
                all_self_attns = all_self_attns + (attn_weights,)

        x = self.norm_f(x)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            assert all_hidden_states is not None  # pyright
            all_hidden_states = all_hidden_states + (x,)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
