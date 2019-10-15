import json

if __name__ == "__main__":
    with open("data/dev.json") as f:
        dev_json = json.load(f)
    dev_tables = set()
    for dev_json_el in dev_json:
        dev_tables.add(dev_json_el["db_id"])
    with open("data/tables.json", "r") as f:
        tables = json.load(f)
    schema_csv = open("schema.csv", "w")

    schema_csv.write("DB ID, TABLE, COLUMN, TYPE, FOREIGN KEY OF\n")
    for table in tables:
        if table["db_id"] not in dev_tables:
            continue
        prev_tab_id = -1
        for col_id in range(1, len(table["column_names_original"])):
            if col_id == 1:
                schema_csv.write("{}, ".format(table["db_id"]))
            else:
                schema_csv.write(", ")
            table_id = table["column_names_original"][col_id][0]
            if prev_tab_id != table_id:
                schema_csv.write("{}, ".format(table["table_names_original"][table_id]))
            else:
                schema_csv.write(", ")
            prev_tab_id = table_id
            schema_csv.write("{}, ".format(table["column_names_original"][col_id][1]))
            schema_csv.write("{}, ".format(table["column_types"][col_id]))
            printed = False
            for f, p in table["foreign_keys"]:
                if f == col_id:
                    table_id = table["column_names_original"][p][0]
                    schema_csv.write("{}.{} ".format(table["table_names_original"][table_id], table["column_names_original"][p][1]))
                    printed = True
                    break
            schema_csv.write("\n")
        schema_csv.write(",,,,\n")
