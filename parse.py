import os
import json
import argparse
from src.rule import semQL
from src.rule.semQL import *


'''
By Each Epoch

    Action ID:
     - Total
     - By Length
    Action Type:
     - Total
     - By Length

        Items:
         - correct / total
         - Chosen answer when wrong
         - List of data that this appears (File name by NL)
'''
# -- Log File Format (Example) --
''' 
Epoch: 0
Type: Sketch
Path History: []
Correct: True 
Prediction: 3
Solution: 3
Solution Action Id: 39
Solution Action Type Str: <class 'src.rule.semQL.Root1'>
Solution Action Type Id: 6
Scores: tensor([-31.5154, -37.8025, -31.0521,   0.0000])

'''

'''
-- Parsed Json Format --
- Epoch:
    - 'prod':
        - length:
            - acc
            - correct
            - total
            - item
    - 'type':
        - length:
            - acc
            - correct
            - total
            - item
'''

# Get grammar
grammar = semQL.Grammar()

# Add C, T To prod2id
grammar.prod2id['C'] = len(grammar.prod2id.keys())
grammar.prod2id['T'] = len(grammar.prod2id.keys())

# Create Backword Dic
id2prod = {y: x for x, y in grammar.prod2id.items()}
id2type = {y: x for x, y in grammar.type2id.items()}

# Create offset Dic
relative_line_idx = ['Type: ', 'Path History: ', 'Correct: ', 'Prediction: ', 'Solution: ', 'Solution Action Id: ',
                     'Solution Action Type Str: ', 'Solution Action Type Id: ', 'Scores: ']
relative_line_idx = {y: x for x, y in enumerate(relative_line_idx)}

non_sketch_classes = [C, T, A]

def create_stack_element(cnt, text):
    element = []
    text = text.split(' ')
    for x in text[1:]:
        if x not in Keywords:
            rule_type = eval(x)
            if rule_type not in non_sketch_classes:
                element += [rule_type]

    return len(element) == 0, (cnt, [eval(text[0])] + element[1:])


def parse_line(lines, idx, feature_name):
    # Get line
    offset = relative_line_idx[feature_name]
    line = lines[idx+offset].strip('\n')

    # Check Assertion
    assert feature_name in line, 'Line:{} Feature_Name:{}'.format(line, feature_name)

    # Get Value only
    return line.replace(feature_name, '')


def parse_epoch(file_name, sql, epoch, lines):
    node_list = []
    element_cnt = 1
    stack = []
    # Parse infos in this epoch
    if 'Solution Action Id: 36' in lines or 'Solution Action Id: 37' in lines or 'Solution Action Id: 38' in lines:
        print('here')
    for idx in range(0, len(lines), 10):
        node = {'epoch': epoch, 'file_name': file_name, 'sql': sql}

        # Meta Info
        # Type
        line = parse_line(lines, idx, 'Type: ')
        node['type'] = line

        line = parse_line(lines, idx, 'Solution: ')
        solution = int(line)
        node['solution'] = solution

        # Path Length:
        line = parse_line(lines, idx, 'Path History: ')
        length = line.count(',')
        node['path_length'] = length if '[]' in line else length + 1

        # Action Type
        line = parse_line(lines, idx, 'Solution Action Type Id: ')
        action_type_id = int(line)
        node['action_type_id'] = action_type_id

        # Action ID
        line = parse_line(lines, idx, 'Solution Action Id: ')
        if 'None' not in line:
            action_id = int(line)
            node['action_id'] = action_id
        else:
            line = parse_line(lines, idx, 'Solution Action Type Str: ')
            line = line.split('.')[-1][:-2]
            assert 'C' in line or 'T' in line
            node['action_id'] = grammar.prod2id[line]

        # Items Info
        # Correctness
        line = parse_line(lines, idx, 'Correct: ')
        correctness = True if 'True' in line else False
        node['correct'] = correctness

        # Predicted Answer
        line = parse_line(lines, idx, 'Prediction: ')
        prediction = int(line)
        node['prediction'] = prediction

        if node['type'] == 'Sketch':
            # Create stack element
            action_type = eval(parse_line(lines, idx, 'Solution Action Type Str: ').split('.')[-1][:-2])
            solution = parse_line(lines, idx, 'Solution: ')
            action = action_type.grammar_dict[int(solution)]
            is_terminal, element = create_stack_element(element_cnt, action)

            # Get Number
            node['length_from_parent'] = element_cnt - stack[-1][0] if stack else 1
            element_cnt += 1

            # Push
            stack += [element]

            # Modify
            if is_terminal:
                while stack:
                    top = stack[-1]
                    if len(top[1]) == 1:
                        del stack[-1]
                        continue
                    else:
                        del top[1][1]
                        break

        # Add to list
        node_list += [node]

    return node_list


def parse_file(f_path):
    file = open(f_path)
    lines = file.readlines()

    all_nodes = []
    # Get All Beginning lines for epoch
    item_begin_lines = []
    for line_num, line in enumerate(lines):
        if 'SQL: ' in line:
            sql = line.split(':')[1]
        if 'Epoch: ' in line:
            line = line.replace('Epoch: ', '').strip('\n')
            epoch_num = int(line)
            item_begin_lines += [(epoch_num, line_num)]

    # Parse All Epoch
    for idx in range(len(item_begin_lines)):
        epoch_num = item_begin_lines[idx][0]
        begin_line_num = item_begin_lines[idx][1] + 1
        end_line_num = len(lines) if idx+1 == len(item_begin_lines) else item_begin_lines[idx+1][1]
        file_name = f_path.split('/')[-1]
        # Parse Epoch
        node_list = parse_epoch(file_name, sql, epoch_num, lines[begin_line_num:end_line_num])
        all_nodes += node_list

    file.close()
    return all_nodes


def parse_list2dic(e_list, key_name, key_dic, length_key):
    # Create prod Dic
    dic_action_id = {id: [] for id in key_dic.keys()}

    # Categorize by action id
    for e_item in e_list:
        id = e_item[key_name]
        dic_action_id[id] += [e_item]

    # By Action ID (Epoch -> Action ID)
    for a_id, a_list in dic_action_id.items():

        # Get All Lengths
        all_lengths = set()
        for a_item in a_list:
            if length_key in a_item.keys():
                all_lengths.add(a_item[length_key])

        # Make dic for length
        dic_length = {len: [] for len in all_lengths}

        # Add by length
        for a_item in a_list:
            if length_key in a_item.keys():
                length = a_item[length_key]
                dic_length[length] += [a_item]

        # correct (Epoch -> Action ID -> Length)
        for length, len_list in dic_length.items():

            total_item_cnt = len(len_list)
            correct_cnt = 0
            for item in len_list:
                correctness = item['correct']
                correct_cnt = correct_cnt + 1 if correctness else correct_cnt

            # Tmp
            tmp = {}
            tmp['correct_cnt'] = correct_cnt
            tmp['total'] = total_item_cnt
            tmp['acc'] = correct_cnt / total_item_cnt
            tmp['items'] = len_list

            # Replace
            dic_length[length] = tmp

        # Replace
        dic_action_id[a_id] = dic_length

    return dic_action_id


def parse_log2json(log_path):
    # Read Files
    onlyfiles = [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]

    # Get all info from files
    total = []
    for f_name in onlyfiles:
        f_path = os.path.join(log_path, f_name)
        parsed_items = parse_file(f_path)
        total += parsed_items

    print("Total items: ", len(total))

    # Get all epochs
    all_epoch = set()
    for item in total:
        all_epoch.add(item['epoch'])

    # Create Epoch Dic
    dic_epoch = {e: [] for e in all_epoch}

    # Categorize by epoch
    for item in total:
        dic_epoch[item['epoch']] += [item]

    # By Epoch (Epoch)
    for epoch, e_list in dic_epoch.items():

        dic_length = parse_list2dic(e_list, 'action_type_id', id2type, 'length_from_parent')
        dic_action_id = parse_list2dic(e_list, 'action_id', id2prod, 'path_length')
        y = []
        for x in e_list:
            if x['action_id'] == 9 and x['correct'] == False:
                y.append(x)

        dic_action_type_id = parse_list2dic(e_list, 'action_type_id', id2type, 'path_length')

        # Replace
        tmp = {'prod': dic_action_id, 'type': dic_action_type_id, 'length_from_parent': dic_length}
        dic_epoch[epoch] = tmp

    print('Parsing Done!')

    out_file_name = 'analysis_out.txt'
    out_file_path = os.path.join(log_path, out_file_name)
    with open(out_file_path, 'w') as f:
        json.dump(dic_epoch, f)

    print('Dumping result to {}'.format(out_file_path))

    return dic_epoch


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log_path', default='./analysis_backup', type=str, help='Path for log directory')
    arg_parser.add_argument('--target_epoch', default='0', type=str, help='Target Epoch to analyze')
    arg_parser.add_argument('--analyze_type', default='type', type=str, help='Analzying Type. i.e. "type" or "prod"')
    args = arg_parser.parse_args()

    # check if file exist
    onlyfiles = [f for f in os.listdir(args.log_path) if os.path.isfile(os.path.join(args.log_path, f))]
    out_file_name = 'analysis_out.txt'
    if out_file_name in onlyfiles:
        print('Loading from file...')
        with open(os.path.join(args.log_path, out_file_name)) as f:
            js = json.load(f)
    else:
        print('Parsing...')
        js = parse_log2json(args.log_path)

    # Analyze
    target_epoch = args.target_epoch if out_file_name in onlyfiles else int(args.target_epoch)
    ana_type = args.analyze_type
    data = js[target_epoch][ana_type]

    if ana_type in ['type', 'length_from_parent']:
        id_dic = id2type
    elif ana_type in ['prod']:
        id_dic = id2prod
    else:
        print('unknown analyze type: ', ana_type)
        exit(-1)

    print('Epoch: {}'.format(target_epoch))
    for idx, action_item in data.items():
        action = id_dic[int(idx)]
        print('\tAction: {}'.format(action))

        total_acc_num = 0
        total_num = 0
        for length, item in action_item.items():
            total_acc_num += item['total'] * item['acc']
            total_num += item['total']
            print('\t\t\tLength: {} Total:{} Acc:{}'.format(length, item['total'], item['acc']))
        if total_num != 0:
            print('\t total: {} acc: {} wrong_num: {}'.format(total_num, total_acc_num / total_num, total_num - total_acc_num))
        else:
            print('\t total: 0')

    print('\nDone..!')
