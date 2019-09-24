import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
import torch.nn.parallel
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

import torch.cuda.comm as comm


class Reduce(Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


def parallel_train(model, dataloader, gpu_num):
    total_loss = 0.0
    model.train()
    dataloader.shuffle()
    batches = dataloader.get_train()
    replicas = torch.nn.parallel.replicate(model.generator, range(gpu_num))
    for batch in batches:
        # preprocess
        input_data, gt_data = model.preprocess(batch)
        if input_data is None:
            continue

        q_embs, q_len, q_ranges_list, table_ranges_list, column_ranges_list, new_schemas, parse_trees = input_data

        new_q_embs = torch.nn.parallel.scatter(q_embs, list(range(gpu_num)))
        batch_ranges = torch.tensor(range(len(q_len)))
        batch_ranges = torch.nn.parallel.scatter(batch_ranges, list(range(gpu_num)))

        def to_batch(list_of_something, batch_ranges):
            new_list = []
            for range in batch_ranges:
                range = range.cpu().numpy()
                batch = []
                for idx in range:
                    batch.append(list_of_something[idx])
                new_list.append(batch)
            return tuple(new_list)

        new_q_len = to_batch(q_len, batch_ranges)
        new_q_ranges_list = to_batch(q_ranges_list, batch_ranges)
        new_table_ranges_list = to_batch(table_ranges_list, batch_ranges)
        new_column_ranges_list = to_batch(column_ranges_list, batch_ranges)
        new_new_schemas = to_batch(new_schemas, batch_ranges)
        new_parse_trees = to_batch(parse_trees, batch_ranges)
        new_input = tuple(zip(new_q_embs, new_q_len, new_q_ranges_list, new_table_ranges_list, new_column_ranges_list,
                        new_new_schemas, new_parse_trees))
        if len(new_input) != 8:
            print(len(batch_ranges))
        outputs = nn.parallel.parallel_apply(replicas, new_input)
        new_score = []
        new_subtrees = []
        for score, subtrees in outputs:
            new_score += score
            new_subtrees += subtrees


        # criterion
        losses, accs = model.loss(new_score, gt_data, new_subtrees)
        # losses = nn.parallel.gather(losses, 0)
        loss = Reduce.apply(*losses)
        # losses = sum(losses)

        # backward
        model.zero_grad()
        loss.backward()
        # losses.backward()
        model.step()
        total_loss += loss.data.cpu().numpy() * len(batch)
    return total_loss / dataloader.get_train_len()


def train(model, dataloader):
    total_loss = 0.0
    model.train()
    dataloader.shuffle()
    batches = dataloader.get_train()
    for batch in batches:
        # preprocess
        input_data, gt_data = model.preprocess(batch)
        if input_data is None:
            continue

        # forward
        score, subtrees = model.forward(input_data)

        # criterion
        loss, accs = model.loss(score, gt_data, subtrees)
        total_loss += loss.data.cpu().numpy() * len(batch)

        # backward
        model.zero_grad()
        loss.backward()
        model.step()
    return total_loss / dataloader.get_train_len()


# Evaluation during training
def eval(model, dataloader, log=False):
    total_acc = np.zeros(model.acc_num)
    total_topk_acc = np.zeros(model.acc_num)
    total_topk_cor_acc = np.zeros(model.acc_num)
    total_topk_in_acc = np.zeros(model.acc_num)
    total_loss = 0.0
    model.eval()
    dataloader.shuffle()
    batches = dataloader.get_eval()
    for idx, batch in enumerate(batches):

        # preprocess
        input_data, gt_data = model.preprocess(batch)

        # forward
        score, subtrees = model.forward(input_data)

        loss, accs = model.loss(score, gt_data, subtrees)
        total_acc += accs
        total_loss += loss.data.cpu().numpy() * len(batch)
        # # Generate Query
        # acc, topk_acc, topk_cor_acc, topk_in_acc = model.evaluate(score, gt_data, batch, log=log)
        #
        # total_acc += acc
        #
        # total_topk_acc += topk_acc
        # total_topk_cor_acc += topk_cor_acc
        # total_topk_in_acc += topk_in_acc

    print("acc: {}".format(total_acc / dataloader.get_eval_len()))
    return total_topk_in_acc / dataloader.get_eval_len(), total_loss / dataloader.get_eval_len()


def test(model, dataloader, output_path):
    file = open(output_path, "w")
    model.eval()
    batches = dataloader.get_eval()
    for batch in batches:
        input_data, gt_data = model.preprocess(batch)

        score = model.forward(input_data)

        gen_sqls = model.gen_sql(score, gt_data)

        for sql in gen_sqls:
            file.write("{}\n".format(sql))
    file.close()


def SIZE_CHECK(tensor, size):
    for idx, dim in enumerate(size):
        if dim is None or dim == -1:
            size[idx] = list(tensor.size())[idx]
    if list(tensor.size()) != size:
        raise AssertionError("{} not match {}".format(list(tensor.size()), size))


def seq_conditional_weighted_num(attention_layer, predicate_tensor, predicate_len, conditional_tensor,
                                 conditional_len=None):
    max_predicate_len = max(predicate_len)
    if conditional_len is not None:
        max_conditional_len = max(conditional_len)
    else:
        max_conditional_len = None
    B = len(predicate_len)
    SIZE_CHECK(predicate_tensor, [B, max_predicate_len, None])
    SIZE_CHECK(conditional_tensor, [B, max_conditional_len, None])
    co_attention = torch.bmm(conditional_tensor, attention_layer(predicate_tensor).transpose(1, 2))
    SIZE_CHECK(co_attention, [B, max_conditional_len, max_predicate_len])
    for idx, num in enumerate(predicate_len):
        if num < max_predicate_len:
            co_attention[idx, :, num:] = -100
    if conditional_len is not None:
        for idx, num in enumerate(conditional_len):
            if num < max_conditional_len:
                co_attention[idx, num:, :] = -100
    softmaxed_attention = F.softmax(co_attention.view(-1, max_predicate_len), dim=1)\
        .view(B, -1, max_predicate_len)
    weighted = (predicate_tensor.unsqueeze(1) * softmaxed_attention.unsqueeze(3)).sum(2)
    SIZE_CHECK(weighted, [B, None, None])
    return weighted


def plain_conditional_weighted_num(att, predicate_tensor, predicate_len, conditional_tensor):
    max_predicate_len = max(predicate_len)
    B = len(predicate_len)

    SIZE_CHECK(predicate_tensor, [B, max_predicate_len, None])
    SIZE_CHECK(conditional_tensor, [B, None])

    co_attention = torch.bmm(conditional_tensor.unsqueeze(1), att(predicate_tensor).transpose(1, 2))\
        .view(B, max_predicate_len)
    for idx, num in enumerate(predicate_len):
        if num < max_predicate_len:
            co_attention[idx, num:] = -100
    co_attention = F.softmax(co_attention, dim=1)
    weighted = (predicate_tensor * co_attention.unsqueeze(2))
    weighted = weighted.sum(1)
    return weighted


def encode_question(bert, inp, inp_len):
    [batch_num, max_seq_len] = list(inp.size())
    mask = np.zeros((batch_num, max_seq_len), dtype=np.float32)
    for idx, len in enumerate(inp_len):
        mask[idx, :len] = np.ones(len, dtype=np.float32)
    mask = torch.LongTensor(mask)
    if torch.cuda.is_available():
        mask = mask.cuda(0)
    encoded, _ = bert(input_ids=inp, attention_mask=mask)
    return encoded[-1]


def run_lstm(lstm, inp, inp_len, hidden=None):
    # Run the LSTM using packed sequence.
    # This requires to first sort the input according to its length.
    sort_perm = np.array(sorted(list(range(len(inp_len))),
        key=lambda k:inp_len[k], reverse=True))
    sort_inp_len = inp_len[sort_perm]
    sort_perm_inv = np.argsort(sort_perm)
    if inp.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda(0)
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda(0)

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],
            sort_inp_len, batch_first=True)
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    #Encode the columns.
    #The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda(0)

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len


def col_tab_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    #Encode the columns.
    #The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda(0)

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len
