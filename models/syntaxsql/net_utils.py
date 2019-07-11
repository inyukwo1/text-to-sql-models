def to_batch_seq(batch):
    q_seq = []
    history = []
    for item in batch:
        q_seq.append(item['question_toks'])
        history.append(item["history"])
    return q_seq, history


# CHANGED
def to_batch_tables(batch, table_type):
    # col_lens = []
    col_seq = []
    tname_seqs = []
    for item in batch:
        tname_toks = [x.split(" ") for x in item["db"]["table_names"]]
        cols = [x.split(" ") for xid, x in item["db"]["column_names"]]
        tab_seq = [xid for xid, x in item["db"]["column_names"]]
        cols_add = []
        for tid, col in zip(tab_seq, cols):
            col_one = []
            col_one.extend(col)
            cols_add.append(col_one)
        col_seq.append(cols_add)
        tname_seqs.append(tname_toks)

    return col_seq, tname_seqs


def to_batch_from_candidates(par_tab_nums, batch):
    from_candidates = []
    for idx, item in enumerate(batch):
        table_candidate = item["from"]
        col_candidates = [0]
        for col, par in enumerate(par_tab_nums[idx]):
            if str(par) in table_candidate:
                col_candidates.append(col)
        from_candidates.append(col_candidates)

    return from_candidates


def make_compound_table(dev_db_compound_num, table_dict, my_db_id, db_ids):
    if dev_db_compound_num == 0:
        return table_dict[my_db_id]
    selected_db_ids = random.sample(db_ids, dev_db_compound_num)
    if my_db_id in selected_db_ids:
        selected_db_ids.remove(my_db_id)

    compound_table = deepcopy(table_dict[my_db_id])
    for dev_db_id in selected_db_ids:
        new_table = table_dict[dev_db_id]
        if random.randint(0, 10) < 5:
            new_table = compound_table
            compound_table = deepcopy(table_dict[dev_db_id])
        compound_table = append_table(compound_table, new_table)
    return compound_table

def append_table(compound_table, new_table):
    for table_name in new_table["table_names"]:
        if table_name in compound_table["table_names"]:
            return compound_table
    new_table_offset = len(compound_table["table_names"])
    new_column_offset = len(compound_table["column_names"]) - 1
    compound_table["table_names"].extend(new_table["table_names"])
    compound_table["table_names_original"].extend(new_table["table_names_original"])
    for p in new_table["primary_keys"]:
        compound_table["primary_keys"].append(p + new_column_offset)
    for f, p in new_table["foreign_keys"]:
        compound_table["foreign_keys"].append([f + new_column_offset, p + new_column_offset])
    compound_table["column_types"].extend(new_table["column_types"])
    for t, name in new_table["column_names_original"][1:]:
        compound_table["column_names_original"].append([t + new_table_offset, name])
    for t, name in new_table["column_names"][1:]:
        compound_table["column_names"].append([t + new_table_offset, name])
    return compound_table

def index_to_column_name(index, table):
    column_name = table["column_names"][index][1]
    table_index = table["column_names"][index][0]
    table_name = table["table_names"][table_index]
    return table_name, column_name, index

