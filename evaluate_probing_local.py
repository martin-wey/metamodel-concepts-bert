"""
This script is used to evaluate the probing ability of a RoBERTa language model.
It replaces token of interest by a <mask> token and attempts to predict the ground truth.
Reported metrics: Exact match, Recall@k, MRR@k and execution time.
"""

from transformers import RobertaForMaskedLM, RobertaTokenizerFast
import torch
from collections import Counter
from tqdm import tqdm
import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='Path to the pretrained RoBERTa model.'
    )
    parser.add_argument(
        '--tokenizer_path', type=str, default=None,
        help='Path to the BPE tokenizer.'
    )
    parser.add_argument(
        '--test_file', type=str, default=None,
        help='The test data file (a text file).'
    )
    parser.add_argument(
        '--pred_type', type=str, default=None,
        help='Prediction type (cls=classifiers, attrs=attributes, assocs=associations)'
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level="INFO"
    )

    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path)
    model = RobertaForMaskedLM.from_pretrained(args.model_path).cuda()

    with open(args.test_file) as fin:
        data = fin.readlines()

    # test_type = 'cls' or 'attrs' or 'assocs'
    test_type = args.pred_type
    open_char = '('
    close_char = ')'

    # count the occurrences of the recommendations
    recommendations_count = Counter()

    n_test = 0
    accuracies = {'1': 0, '5': 0, '10': 0, '20': 0}
    mrrs = {'1': 0, '5': 0, '10': 0, '20': 0}
    start_time = time.time()
    for idx in tqdm(range(len(data))):
        if data[idx] != '\n':
            test_sample = data[idx].split(';')
            context = test_sample[0].split()
            current_test_cls = test_sample[1].strip()

            last_type = None
            evaluate = False
            for i, tkn in enumerate(context):
                # check that the current classifier is the one we need to evaluate
                if context[i] == '<NAME>':
                    last_type = 'cls'
                    if context[i + 1] == current_test_cls:
                        evaluate = True
                    else:
                        evaluate = False
                elif context[i] == '<ATTRS>':
                    last_type = 'attrs'
                elif context[i] == '<ASSOCS>':
                    last_type = 'assocs'

                if evaluate and last_type == test_type:
                    test_idx = -1
                    if (test_type == 'attrs' or test_type == 'assocs') and \
                            context[i - 2] == open_char and context[i + 1] == close_char:
                        test_idx = i
                    elif test_type == 'cls' and context[i - 1] == '<NAME>':
                        test_idx = i

                    if test_idx != -1:
                        ground_truth = context[test_idx]
                        context[test_idx] = tokenizer.mask_token
                        test_input = ' '.join(context)

                        input = tokenizer.encode(test_input, return_tensors='pt').cuda()
                        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

                        token_logits = model(input)[0]
                        mask_token_logits = token_logits[0, mask_token_index, :]

                        top_k_tokens = torch.topk(mask_token_logits, 20, dim=1).indices[0].tolist()

                        found = False
                        for idx, token in enumerate(top_k_tokens):
                            prediction = tokenizer.decode([token]).strip()
                            if prediction == ground_truth:
                                for k in [1, 5, 10, 20]:
                                    accuracies[str(k)] += 1 if k > idx else 0
                                    mrrs[str(k)] += 1 / (idx + 1) if k > idx else 0
                                found = True
                            recommendations_count.update([prediction])

                        n_test += 1
                        # remove mask and restore current test sample
                        context[test_idx] = ground_truth

    end_time = time.time()
    logger.info(f'Total execution time: {round(end_time - start_time, 2)} seconds')
    logger.info(f'Execution time per sample: {round((end_time - start_time) / n_test, 2)} seconds')

    # get metrics in percentage
    for k in ['1', '5', '10', '20']:
        accuracies[k] = round(accuracies[k] / n_test, 4)
        mrrs[k] = round(mrrs[k] / n_test, 4)

    logger.info(f'***** Test results  *****')
    logger.info(f'Number of test samples: {n_test}')
    logger.info(f'R@k: {accuracies}')
    logger.info(f'MRR@k: {mrrs}')
    logger.info(f'Most common suggestions" {recommendations_count.most_common(10)}')


if __name__ == '__main__':
    main()
