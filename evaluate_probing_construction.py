from collections import Counter
import argparse
import logging
import sys
import time
from pprint import pprint

from transformers import RobertaForMaskedLM, RobertaTokenizerFast
import torch
from tqdm import tqdm


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

    # count the occurrences of the recommendations
    recommendations_count = Counter()

    n_test = 0
    accuracies = {}
    mrrs = {}
    for pred_type in ['cls', 'attrs', 'assocs']:
        accuracies[pred_type] = {}
        mrrs[pred_type] = {}
        for i in range(0, 81, 10):
            accuracies[pred_type][str(i)] = {'1': 0, '5': 0, '10': 0, '20': 0, 'n_test': 0}
            mrrs[pred_type][str(i)] = {'1': 0, '5': 0, '10': 0, '20': 0, 'n_test': 0}

    start_time = time.time()
    for idx in tqdm(range(len(data[:100]))):
        if data[idx] != '\n':
            sample_data = data[idx].split(';')

            context_size = int(sample_data[1])
            if context_size > 80:
                context_intv_idx = str(80)
            else:
                context_intv_idx = str(((context_size - 1) // 10) * 10)

            test_type = sample_data[2]
            ground_truth = sample_data[3].strip()

            test_input = sample_data[0]

            input = tokenizer.encode(test_input, return_tensors='pt').cuda()
            mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

            token_logits = model(input)[0]
            mask_token_logits = token_logits[0, mask_token_index, :]

            top_k_tokens = torch.topk(mask_token_logits, 20, dim=1).indices[0].tolist()

            logger.debug(f'Ground truth: {ground_truth}')
            logger.debug(f'Suggestions: {[tokenizer.decode([token]).strip() for token in top_k_tokens]}')
            for idx, token in enumerate(top_k_tokens):
                prediction = tokenizer.decode([token]).strip()
                if prediction == ground_truth:
                    for k in [1, 5, 10, 20]:
                        accuracies[test_type][context_intv_idx][str(k)] += 1 if k > idx else 0
                        mrrs[test_type][context_intv_idx][str(k)] += 1 / (idx + 1) if k > idx else 0
                recommendations_count.update([prediction])

            accuracies[test_type][context_intv_idx]['n_test'] += 1
            mrrs[test_type][context_intv_idx]['n_test'] += 1
            n_test += 1

    end_time = time.time()
    logger.info(f'Total execution time: {round(end_time - start_time, 2)} seconds')
    logger.info(f'Execution time per sample: {round((end_time - start_time) / n_test, 2)} seconds')

    for pred_type in ['cls', 'attrs', 'assocs']:
        for i in range(0, 81, 10):
            for k in ['1', '5', '10', '20']:
                if accuracies[pred_type][str(i)]['n_test'] != 0:
                    accuracies[pred_type][str(i)][str(k)] = round(
                        accuracies[pred_type][str(i)][str(k)] / accuracies[pred_type][str(i)]['n_test'], 4
                    )
                    mrrs[pred_type][str(i)][str(k)] = round(
                        mrrs[pred_type][str(i)][str(k)] / mrrs[pred_type][str(i)]['n_test'], 4
                    )

    logger.info(f'***** Test results *****')
    logger.info(f'Number of test samples: {n_test}')
    logger.info(f'R@k: {pprint(accuracies)}')
    logger.info(f'MRR@k: {pprint(mrrs)}')
    logger.info(f'Most common suggestions" {recommendations_count.most_common(10)}')


if __name__ == '__main__':
    main()
