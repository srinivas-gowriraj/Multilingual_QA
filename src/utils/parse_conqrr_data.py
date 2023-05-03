import argparse
import json
from datasets import load_dataset

def main(args):
    # input_data = []
    # with open(args.input_file_path, 'r') as f:
    #     for line in f:
    #         input_data.append(json.loads(line))
    # assert len(input_data) == 2000
    rewritten_query = load_dataset('json', data_files=args.rewritten_query_fp)["train"]
    original_data = load_dataset('json', data_files=args.original_data_fp)["train"]
    assert len(rewritten_query) == len(original_data)
    orig_data = original_data.remove_columns(['query'])
    orig_data = orig_data.add_column('query', rewritten_query['output'])
    orig_data.to_json(args.output_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-rqfp", "--rewritten_query_fp", type=str)
    parser.add_argument("-odfp", "--original_data_fp", type=str)
    parser.add_argument("-ofp", "--output_file_path", type=str)
    args = parser.parse_args()
    main(args)
