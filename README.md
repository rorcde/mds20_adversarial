# mds20_adversarial
This is the repository for the Models of Sequence Data 2020 Edition for the project Generating Natural Language Adversarial Examples on a Large Scale with Generative Models. 

## The goal of the project:
The goal is to create an algorithm for adversarial examples generation for sequences based on transformers and examine a defense against it.

## TODO: 
- Implement the paper:
  - Load and process dataset (using Datasets library for NLP tasks) (Done)
  
  - Use pre-trained BertMaskedLM as language model (Done)
  - Create a dataset for Deep Levenstein model with random pair sampling and sequence modifying (Done)
  - Implement and train Deep Levenstein Model (Done)
  - Implement and train Substitute classifier and Target classifier (Done)
  - Implement Sampling fool attack on toy example (Done)
  - Compose DILMA Model (Done)
  
- Run experiments:
  - Run proposed DILMA attack for TREC dataset (TD)
  
## Important:
- Please note that due to the huge size of the models used in this project, testing can only be done on Google Colab (or at your local PC if you have 4-5 GBs of VRAM :))
- Therefore, to check our results, one may run the `DILMA` notebook from the corresponding folder. Please note that this notebook uses pretrained models that can be downloaded from the `data` folder


## Tasks and requirements:

1) Run the proposed DILMA attack for at least one NLP dataset
2) Measure the attack success rate, perplexity
3) Evaluate the validity rate by manual processing of generated texts
4) Retrain the model to generate adversarial examples without local adaptation to each sequence
5) Search for the universal attack in the embedded space
6) Code should be in AllenNLP, PyTorch. You can use all code you can find.


## Dataset description 

The [Text REtrieval Conference (TREC) Question Classification dataset](https://github.com/huggingface/datasets/blob/master/datasets/trec/trec.py) contains 5500 labeled questions in the training set and another 500 for the test set. The dataset has 6 labels, 47 level-2 labels. The average length of each sentence is 10, with a vocabulary size of 8700.

Data are collected from four sources: 4,500 English questions published by USC (Hovy et al., 2001), about 500 manually constructed questions for a few rare classes, 894 TREC 8 and TREC 9 questions, and 500 questions from TREC 10, which serves as the test set.

## Ideas on the implementation: (by Daniil)
- Models used in the project need to be trained separately as the proposed approach uses a pretrained model. All models are going to be available in the specific folder `models.`
- TREC Dataset used in the paper is available on 
- Specific functions are stored at folder `utils.`
- Pre-trained model weights and created dataset for Deep Lev are stored in ``data. ``
- Examples of usage and experiments will be done in Jupyter Notebooks and stored in folder `examples.`

## Our team 
- Alexander Esin (@aleksandryessin)
- Daniil Moskovskiy (@etomoscow)
- Margarita Sharkova (@margaretshark)
