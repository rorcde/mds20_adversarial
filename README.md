# mds20_adversarial
This is the repository for the Models of Sequence Data 2020 Edition for the project Generating Natural Language Adversarial Examples on a Large Scale with Generative Models. 

## The goal of the project:
The goal is to create an algorithm for adversarial examples generation for sequences based on transformers and examine a defense against it.

## TODO: 
- Implement the paper:
  - Implement and train classification models - DONE (Daniil)
  - Implement and train language model - DONE (Daniil)
  - Implement and train Deep Levenstein Model - Done (Margarita)
  - Implement ST Gumbel-Softmax Estimator - Done (Daniil)
  - Compose DILMA Model - TBD (Daniil / Margarita) 
- Run experiments:
  - Run proposed DILMA attack for TREC dataset - TBD


## Installation and running:
- TBD

## Tasks and requirements:

1) Run the proposed DILMA attack for at least one NLP dataset
2) Measure the attack success rate, perplexity
3) Evaluate the validity rate by manual processing of generated texts
4) Retrain the model to generate adversarial examples without local adaptation to each sequence
5) Search for the universal attack in the embedded space
6) Code should be in AllenNLP, PyTorch. You can use all code you can find.

## Existing implementation:
- Provided by authors of paper using `AlienNLP` framework, testing is done by Alexander. 
- Assumes to use `Docker` for running 

## Ideas on the implementation: (by Daniil)
- Models used in the project need to be trained separately as the proposed approach uses pretrained model. All models are going to be available in the specific folder `models`
- Datasets used in the paper are available at PyTorch
- Specific functions will be stored separately at folder `utils`
- Examples of usage and experiments will be done in Jupyter Notebooks and will be stored at folder `examples`

## Our team 
- Alexander Esin (@aleksandryessin)
- Daniil Moskovskiy (@etomoscow)
- Margarita Sharkova (@margaretshark)
