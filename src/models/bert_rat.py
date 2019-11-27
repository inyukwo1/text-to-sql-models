import torch.nn as nn
import torch
import math
from transformers import *


BERT = (BertModel, BertTokenizer, 'bert-base-uncased', 768)
class BERT_RAT(nn.Module):
    def __init__(self, num_relation):
        super(BERT_RAT, self).__init__()
        model_class, tokenizer_class, pretrained_weight, dim = BERT
        self.bert_encoder = model_class.from_pretrained(pretrained_weight)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        self.key_modules = nn.ParameterList([nn.Parameter(torch.zeros(num_relation, self.bert_encoder.config.hidden_size // self.bert_encoder.config.num_attention_heads))
                                          for _ in range(self.bert_encoder.config.num_hidden_layers)])
        self.value_modules = nn.ParameterList([nn.Parameter(torch.zeros(num_relation, self.bert_encoder.config.hidden_size // self.bert_encoder.config.num_attention_heads))
                                          for _ in range(self.bert_encoder.config.num_hidden_layers)])

    def forward(self, input_ids, relations, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.bert_encoder.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.bert_encoder.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.bert_encoder.config.num_hidden_layers

        embedding_output = self.bert_encoder.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encode(self.bert_encoder.encoder, embedding_output, relations,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert_encoder.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs

    def encode(self, encoder, hidden_states, relations, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(encoder.layer):
            if encoder.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = self.layer(layer_module, i, hidden_states, relations, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if encoder.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if encoder.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if encoder.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if encoder.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def layer(self, layer_module, idx, hidden_states, relations, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(layer_module.attention, idx, hidden_states, relations, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = layer_module.intermediate(attention_output)
        layer_output = layer_module.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

    def attention(self, attention_module, idx, hidden_states, relations, attention_mask, head_mask):
        self_outputs = self.self_attention(attention_module.self, idx, hidden_states, relations, attention_mask, head_mask)
        attention_output = attention_module.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def self_attention(self, self_attention_module, idx, hidden_states, relations, attention_mask, head_mask):
        mixed_query_layer = self_attention_module.query(hidden_states)
        mixed_key_layer = self_attention_module.key(hidden_states)
        mixed_value_layer = self_attention_module.value(hidden_states)

        query_layer = self_attention_module.transpose_for_scores(mixed_query_layer)
        key_layer = self_attention_module.transpose_for_scores(mixed_key_layer)
        value_layer = self_attention_module.transpose_for_scores(mixed_value_layer)

        B, relation_shape_x, relation_shape_y = relations.size()
        relation_indices = relations.view(-1)
        # key_relations = torch.index_select(self.key_modules[idx], 0, relation_indices).view(B, relation_shape_x, relation_shape_y, self_attention_module.num_attention_heads, self_attention_module.attention_head_size)
        # key_relations = key_relations.permute(0, 3, 1, 2, 4)
        #
        # value_relations = torch.index_select(self.value_modules[idx], 0, relation_indices).view(B, relation_shape_x, relation_shape_y, self_attention_module.num_attention_heads, self_attention_module.attention_head_size)
        # value_relations = value_relations.permute(0, 3, 1, 2, 4)
        # key_relation_ans = torch.matmul(query_layer.unsqueeze(3).cpu(), key_relations.transpose(-1, -2).cpu()).squeeze(3)

        key_relations = torch.index_select(self.key_modules[idx], 0, relation_indices).view(B, relation_shape_x,
                                                                                            relation_shape_y,
                                                                                            self_attention_module.attention_head_size)

        value_relations = torch.index_select(self.value_modules[idx], 0, relation_indices).view(B, relation_shape_x,
                                                                                                relation_shape_y,
                                                                                                self_attention_module.attention_head_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if 5 < idx < 7:
            key_relation_anses = []
            # query_layer: (B, num_attention_heads, i, attention_head_size)
            for head in range(self_attention_module.num_attention_heads):
                key_relation_anses.append(torch.matmul(query_layer[:,head,:,:].unsqueeze(2), key_relations.transpose(-1, -2)).squeeze(2))
            key_relation_ans = torch.stack(key_relation_anses, dim=1)

            attention_scores = attention_scores + key_relation_ans

        attention_scores = attention_scores / math.sqrt(self_attention_module.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self_attention_module.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # attention_probs: (B, num_attention_heads, i, j)
        # value_layer: (B, num_attention_heads, i, attention_head_size)
        # value_relations: (B, num_attention_heads, i, j, attention_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # value_relation_ans = torch.matmul(attention_probs.cpu().unsqueeze(3), value_relations.cpu()).squeeze(3)
        if 5 < idx < 7:
            value_relation_anses = []
            for head in range(self_attention_module.num_attention_heads):
                value_relation_anses.append(torch.matmul(attention_probs[:,head,:,:].unsqueeze(2), value_relations).squeeze(2))
            value_relation_ans = torch.stack(value_relation_anses, dim=1)
            context_layer = context_layer + value_relation_ans

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self_attention_module.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self_attention_module.output_attentions else (context_layer,)
        return outputs
