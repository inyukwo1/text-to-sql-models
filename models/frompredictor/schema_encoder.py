import torch
import torch.nn
from typing import List
from datasets.schema import Schema


class SchemaEncoder(torch.nn.Module):
    def __init__(self, H_PARAM):
        super(SchemaEncoder, self).__init__()
        # TODO - [experiments]
        #          no_entity <--
        #          how to do else?
        self.no_entity = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.no_entity.requires_grad = True

        # TODO - [experiments]
        #          nothing
        #          sum
        #          gru_cell <--
        #          gnn
        self.table_col_gru = torch.nn.GRUCell(H_PARAM["encoded_num"], H_PARAM["encoded_num"])
        self.table_gru = torch.nn.GRUCell(H_PARAM["encoded_num"], H_PARAM["encoded_num"])

        self.table_type = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.table_type.requires_grad = True
        self.foreign_type = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.foreign_type.requires_grad = True
        self.primary_type = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.primary_type.requires_grad = True
        self.text_type = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.text_type.requires_grad = True
        self.number_type = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.number_type.requires_grad = True
        self.time_type = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.time_type.requires_grad = True
        self.boolean_type = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.boolean_type.requires_grad = True
        self.others_type = torch.nn.Parameter(torch.randn(H_PARAM["encoded_num"]))
        self.others_type.requires_grad = True

    def forward(self, schema: Schema, table_tensors: List[torch.Tensor], col_tensors: List[torch.Tensor]):
        return torch.stack(table_tensors), torch.stack(col_tensors), self.no_entity
        new_table_tensors = torch.stack(table_tensors)
        new_table_tensors = self.table_gru(new_table_tensors, self.table_type.repeat(len(table_tensors), 1))
        col_aligned_table_tensors = []
        for tab_id in range(schema.tab_num()):
            col_aligned_table_tensors += [table_tensors[tab_id]] * len(schema.get_child_col_ids(tab_id))
        col_aligned_table_tensors = torch.stack(col_aligned_table_tensors)
        type_tensors = []
        for col_id in schema.get_all_col_ids():
            if schema.get_col_type(col_id) == "foreign":
                type_tensors.append(self.foreign_type)
            if schema.get_col_type(col_id) == "primary":
                type_tensors.append(self.primary_type)
            if schema.get_col_type(col_id) == "text":
                type_tensors.append(self.text_type)
            if schema.get_col_type(col_id) == "number":
                type_tensors.append(self.number_type)
            if schema.get_col_type(col_id) == "time":
                type_tensors.append(self.time_type)
            if schema.get_col_type(col_id) == "boolean":
                type_tensors.append(self.boolean_type)
            if schema.get_col_type(col_id) == "others":
                type_tensors.append(self.others_type)
        type_tensors = torch.stack(type_tensors)
        col_tensors = torch.stack(col_tensors)
        new_col_tensors = self.table_col_gru(col_tensors, self.table_col_gru(col_aligned_table_tensors, type_tensors))
        return new_table_tensors, new_col_tensors, self.no_entity
