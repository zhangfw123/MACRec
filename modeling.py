
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)
import torch
from torch import nn
import copy
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss
import ipdb
from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def create_contrastive_model(config):
    return CrossModalContrastive(config)
class baseT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

class CrossModalContrastive(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):
        super().__init__(config)


        self.temperature = 0.1
        self.contrastive_weight = 0.01  

    def get_encoder_embeddings(self, input_ids, attention_mask=None):

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state

    def contrastive_loss(self, model_a_embeddings, model_b_embeddings, batch_size):

        model_a_norm = F.normalize(model_a_embeddings, dim=-1)
        model_b_norm = F.normalize(model_b_embeddings, dim=-1)
        
        logits = torch.matmul(model_a_norm, model_b_norm.transpose(-2, -1)) / self.temperature
        

        labels = torch.arange(batch_size, device=logits.device)
        

        loss_contrastive = F.cross_entropy(logits, labels)
        
        return loss_contrastive


    def total_loss(self, lm_logits, labels, decoder_input_ids, is_contrastive_task=False, 
                   text_embeddings=None, image_embeddings=None):


        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)

            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


        contrastive_loss = torch.tensor(0.0, device=lm_logits.device, dtype=lm_logits.dtype)
        if is_contrastive_task and text_embeddings is not None and image_embeddings is not None:
            batch_size = text_embeddings.size(0)
            contrastive_loss = self.contrastive_loss(text_embeddings, image_embeddings, batch_size)
        total_loss = loss + self.contrastive_weight * contrastive_loss
        return total_loss

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,
        return_hidden_state=False,
        task_flag=None, 
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        mask = (task_flag == 1)
        if mask is not None:
            contrastive_ids = input_ids[mask]
            contrastive_attention_mask = attention_mask[mask]
        else:
            contrastive_ids = None
            contrastive_attention_mask = None

        if encoder_outputs is None:
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
        is_contrastive_task = False

        model_a_embeddings  = None
        model_b_embeddings = None
        
        if  contrastive_ids.shape[0] > 0:
            threshold = 32100
            eos_token_id = 1

            contrastive_ids_processed = []
            for i in range(contrastive_ids.shape[0]):

                valid = contrastive_ids[i][contrastive_ids[i] >= threshold]


                valid = torch.cat([valid, torch.tensor([eos_token_id], device=contrastive_ids.device)])
                contrastive_ids_processed.append(valid)
            contrastive_ids_processed = torch.stack(contrastive_ids_processed)
            contrastive_attention_mask_processed = (contrastive_ids_processed != 0).long()


            model_a_embeddings = self.get_encoder_embeddings(contrastive_ids_processed, contrastive_attention_mask_processed)

            if contrastive_attention_mask is not None:
                model_a_embeddings = (model_a_embeddings * contrastive_attention_mask_processed.unsqueeze(-1)).sum(dim=1) / contrastive_attention_mask_processed.sum(dim=1, keepdim=True)
            else:
                model_a_embeddings = model_a_embeddings.mean(dim=1)
            

            if labels is not None:

                label_mask = (labels != -100).long()[:, :5]
                labels_proceed = labels[:, :5]
                model_b_embeddings = self.get_encoder_embeddings(labels_proceed, label_mask)

                if label_mask is not None:
                    model_b_embeddings = (model_b_embeddings * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1, keepdim=True)
                else:
                    model_b_embeddings = model_b_embeddings.mean(dim=1)
            is_contrastive_task = True

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)


        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)


        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]


        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        

        loss = self.total_loss(
            lm_logits, labels, decoder_input_ids, 
            is_contrastive_task, model_a_embeddings, model_b_embeddings
        )


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