import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    '''
    should consider consistent output. (only returning the first variable)
    '''

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.parameter.Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.parameter.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.parameter.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.parameter.Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.parameter.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.parameter.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, relation_k=None, relation_v=None, key_padding_mask=None, need_weights=True, attn_mask=None):
        # relation_k : [batch_size, k_len, k_len, dim_of_head]
        # relation_v : [batch_size, k_len, k_len, dim_of_head]
        assert self._qkv_same_embed_dim, 'query, key, value should all be same size'
        embed_dim_to_check = self.embed_dim
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        bias_k = self.bias_k
        bias_v = self.bias_v
        add_zero_attn = self.add_zero_attn
        dropout_p = self.dropout
        out_proj_weight = self.out_proj.weight
        out_proj_bias = self.out_proj.bias
        training = self.training

        # Modify code from nn.functional.multi_head_attention_forward
        kv_same = torch.equal(key, value)
        qkv_same = torch.equal(query, key) and kv_same

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim_to_check == embed_dim, 'self:{} {}'.format(embed_dim_to_check, embed_dim)
        assert key.size() == value.size()

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        if qkv_same:
            q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif kv_same:
            _start, _end = 0, embed_dim
            _w = in_proj_weight[_start:_end, :]
            _b = in_proj_bias[_start:_end] if in_proj_bias is not None else None
            q = nn.functional.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = v = None
            else:
                _start, _end = embed_dim, None
                _w = in_proj_weight[_start:, :]
                _b = in_proj_bias[_start:] if in_proj_bias is not None else None
                k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            raise NotImplemented('Not implemented yet') # Expected qkv same or kv same
            pass

        q = q * scaling

        assert bias_k is None and bias_v is None

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                              dtype=attn_mask.dtype,
                                                              device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        # q k
        attn_output_weights_k = torch.bmm(q, k.transpose(1, 2))

        if qkv_same and relation_k is not None:
            # q r_k
            r_k = relation_k.repeat(num_heads, 1, 1, 1)
            q_len = q.shape[-2]
            q = q.contiguous().view(bsz*num_heads*q_len, 1, head_dim)
            r_k = r_k.view(bsz*num_heads*q_len, head_dim, q_len)
            attn_output_weights_relation_k = torch.bmm(q, r_k).view(bsz*num_heads, q_len, q_len)

        # Combine
        attn_output_weights = attn_output_weights_k + attn_output_weights_relation_k if qkv_same and relation_k is not None else attn_output_weights_k

        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            assert attn_mask.shape[1] == attn_mask.shape[2]
            q_len = attn_mask.shape[2]
            attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.repeat(num_heads, 1, 1, 1)
            attn_mask = attn_mask.view(bsz*num_heads, q_len, q_len)
            attn_output_weights = attn_output_weights + attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),
                                                                    float('-inf'),)
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        # Remove NAN
        attn_output_weights = torch.where(attn_output_weights == attn_output_weights,
                                          attn_output_weights, torch.zeros(attn_output_weights.shape, dtype=attn_output_weights.dtype, device=attn_output_weights.device))
        attn_output_weights = nn.functional.dropout(attn_output_weights, p=dropout_p, training=training)

        # qk v
        attn_output_v = torch.bmm(attn_output_weights, v)

        # qk v_k
        if qkv_same and relation_v is not None:
            r_v = relation_v.repeat(num_heads, 1, 1, 1)
            q_len = attn_output_weights.shape[-1]
            attn_output_weights = attn_output_weights.view(bsz*num_heads*q_len, 1, q_len)
            r_v = r_v.view(bsz*num_heads*q_len, q_len, head_dim)
            tmp = torch.bmm(attn_output_weights, r_v)
            attn_output_r_v = tmp.view(bsz*num_heads, q_len, head_dim)

        # Combine
        attn_output = attn_output_v + attn_output_r_v if qkv_same and relation_v is not None else attn_output_v

        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        return attn_output