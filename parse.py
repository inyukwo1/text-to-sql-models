import os
import json
import argparse
from src.rule import semQL

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
'''
 -- Log File Format (Example) -- 
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
-- Result Dictionary Format --
- Epoch:
    - prod_id:
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

def parse_line(lines, idx, feature_name):
    # Get line
    offset = relative_line_idx[feature_name]
    line = lines[idx+offset].strip('\n')

    # Check Assertion
    assert feature_name in line, 'Line:{} Feature_Name:{}'.format(line, feature_name)

    # Get Value only
    return line.replace(feature_name, '').replace(' ', '')



def parse_epoch(file_name, epoch, lines):
    node_list = []
    # Parse infos in this epoch
    for idx in range(0, len(lines), 10):
        node = {'epoch': epoch, 'file_name': file_name}

        # Meta Info
        # Type
        line = parse_line(lines, idx, 'Type: ')
        node['type'] = line

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
        node_list = parse_epoch(file_name, epoch_num, lines[begin_line_num:end_line_num])
        all_nodes += node_list

    file.close()
    return all_nodes

def parse_list2dic(e_list, key_name, key_dic):
    # Create prod Dic
    dic_action_id = {id: [] for id in key_dic.keys()}

    # Categorize by action id
    for e_item in e_list:
        #id = e_item['action_id']
        id = e_item[key_name]
        dic_action_id[id] += [e_item]

    # By Action ID (Epoch -> Action ID)
    for a_id, a_list in dic_action_id.items():

        # Get All Lengths
        all_lengths = set()
        for a_item in a_list:
            all_lengths.add(a_item['path_length'])

        # Make dic for length
        dic_length = {len: [] for len in all_lengths}

        # Add by length
        for a_item in a_list:
            length = a_item['path_length']
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

def analyze(log_path):
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

        dic_action_id = parse_list2dic(e_list, 'action_id', id2prod)
        dic_action_type_id = parse_list2dic(e_list, 'action_type_id', id2type)

        # Replace
        tmp = {'prod': dic_action_id, 'type': dic_action_type_id}
        dic_epoch[epoch] = tmp

    print('Parsing Done!')

    out_file_name = 'analysis_out.txt'
    with open(out_file_name, 'w') as f:
        json.dump(dic_epoch, f)

    print('Dumping result to {}'.format(out_file_name))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log_path', default='./analysis_backup', type=str, help='Path for log directory')
    args = arg_parser.parse_args()

    analyze(args.log_path)
