import json
import argparse
from commons.utils import eval, test

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='', help='Model Name')
    parser.add_argument('--data_name', type=str, default='', help='Dataset Name')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    # Load Model
    if args.model_name == 'syntaxsql':
        from models.syntaxsql.syntaxsql import SyntaxSQL as Model
        H_PARAMS = json.loads(open('./models/syntaxsql/parameters.json').read())

    elif args.model_name == 'typesql':
        from models.typesql.typesql import TypeSQL as Model
        H_PARAMS = json.loads(open('./models/typesql/parameters.json').read())

    elif args.model_name == 'sqlnet':
        from models.sqlnet.sqlnet import SQLNet as Model
        H_PARAMS = json.loads(open('./models/sqlnet/parameters.json').read())

    elif args.model_name == 'frompredictor':
        from models.frompredictor.from_predictor import FromPredictor as Model
        H_PARAMS = json.loads(open('./models/frompredictor/parameters.json').read())

    else:
        print('Give correct model name!')
        exit(-1)

    if args.toy == True:
        H_PARAMS['toy'] = True
        H_PARAMS['gpu'] = False
    model = Model(H_PARAMS)
    model.load_model()

    # Load DataLoader
    if args.data_name == 'spider':
        from datasets.spider.data_loader import DataLoader

    else:
        print('Give correct dataset name!')
        exit(-1)

    load_option = H_PARAMS['load_option']if 'load_option' in H_PARAMS.keys() else None
    dataloader = DataLoader(H_PARAMS['batch_size'])
    dataloader.load_data('test', load_option)

    if args.model_name == 'syntaxsql':
        model.set_test_table_dict(dataloader.schemas, dataloader.test_data)
        assert dataloader.batch_size == 1

    # Testing
    if args.model_name == 'frompredictor':
        acc = eval(model, dataloader, log=args.log)
        print('Average Acc:{}'.format(acc))
    else:
        test(model, dataloader, H_PARAMS['output_path'])

    print('Test output is written')