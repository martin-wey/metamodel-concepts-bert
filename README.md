# Learning metamodel concepts with BERT-like language models
### Paper: [Recommending Metamodel Concepts during Modeling Activities with Pre-Trained Language Models](https://arxiv.org/abs/2104.01642)
---
In this repository, you can find all the materials to reproduce the experiments in the aforementioned paper. 
Please, cite the dataset or our work as
```sh
@article{weyssow2021recommending,
  title={Recommending Metamodel Concepts during Modeling Activities with Pre-Trained Language Models},
  author={Weyssow, Martin and Sahraoui, Houari and Syriani, Eugene},
  journal={arXiv preprint arXiv:2104.01642},
  year={2021}
}
```
For further questions and or bugs, feel free to contact us: martin.weyssow@umontreal.ca

## Dependencies
To run the training and evaluation scripts, you need Python>=3.7. 
We use [HuggingFace](https://huggingface.co/) transformer library to train and evaluate our RoBERTa language model. You can find installation instructions here: https://huggingface.co/transformers/installation.html. 
Additionally, we use both [Datasets](https://huggingface.co/docs/datasets/) and [Tokenizers](https://huggingface.co/docs/tokenizers/python/latest/) libraries to load and manipulate our datasets.

## Data acquisition
We make all our data and models available at the following link  https://doi.org/10.5281/zenodo.5579980.

- ```models.7z``` contains our pretrained language model that can be used for inference.
- ```tokenizers.7z``` contains the tokenizers associated with the language model.
- ```train.7z``` contains our training data splitted into a training and validation set.
- ```test.7z``` contains all the test files to reproduce results reported in our evaluation.
- ```train_vocab.txt``` contains the occurrence of each word in the training vocabulary.

Our dataset is based on [MAR dataset]([http://mar-search.org/experiments/models20/](http://mar-search.org/experiments/models20/)) from which we extracted tree-representations of Ecore metamodels. 
We also make our data extraction tool available in the folder ```./ecore-to-tree```.

## Model training and finetuning
If you desire to train a model from scratch or finetune an existing one, you can use the script ```./run_mlm.py```. 
The learning objective used is masked language modeling. However, the script can be easily adapted to other learning objectives such as causal language modeling. 
For instance, to train a RoBERTa langauge model from scratch, you can run:
```sh
python run_mlm.py \
    --run_name ecore-roberta \
    --output_dir ./saved_models/ecore-roberta \
    --train_file ./train/repo_ecore_train.txt \
    --validation_file ./train/repo_ecore_val.txt \
    --num_train_epochs 20 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy=epoch \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --report_to wandb # if you want to report logs to wandb
```
You can find all available training arguments on [HuggingFace's documentation](https://huggingface.co/transformers/main_classes/trainer.html). 
Run ```python run_mlm.py --help``` to show further details about the available arguments. 

## Evaluation
For all the experiments, the script outputs Recall@k and MRR@k effectiveness metrics.
### RQ1 -- global context sampling
To reproduce RQ1, use ```evaluate_probing_global.py``` script. 
```sh
python evaluate_probing_global.py
    --model_path ./models/ecorebert-large \
    --tokenizer_path ./tokenizers/ecorebert-bpe-30k \
    --test_file ./data/test/test_probing_full_context.txt \
    --train_vocab ./data/train_vocab.txt \
    --pred_type cls
```
To run the evaluation for attributes, change ```--pred_type``` argument to ```attrs```. Similarly, to evaluate associations predictions, change its value to ```assocs```.
### RQ2 -- local context sampling
To reproduce RQ2, use ```evaluate_probing_local.py``` script. 
```sh
python evaluate_probing_local.py
    --model_path ./models/ecorebert-large \
    --tokenizer_path ./tokenizers/ecorebert-bpe-30k \
    --test_file ./data/test/test_probing_local_context.txt \
    --pred_type cls
```
### RQ3 -- iterative construction
To reproduce RQ3, use ```evaluate_probing_construction.py``` script. 
```sh
python evaluate_probing_construction.py
    --model_path ./models/ecorebert-large \
    --tokenizer_path ./tokenizers/ecorebert-bpe-30k \
    --test_file ./data/test/test_iterative_construction.txt
```
### RQ4 -- Use cases
To run the evaluation for a specific use case, you can use each script associated to RQ1, RQ2 and RQ3 by specifying its corresponding ```--test_file```. 
For instance, to evaluate the local context sampling strategy with the Java use case:
```
python evaluate_probing_local.py
    --model_path ./models/ecorebert-large \
    --tokenizer_path ./tokenizers/ecorebert-bpe-30k \
    --test_file ./data/test/use_case2/test_probing_local_context.txt
```
