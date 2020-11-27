# Adversarial Attacks on Categorical SequenceClassifiers

## **1  Introduction to adversarial attacks**

Adversarial examples aim at causing target model to make a mistake on prediction. It canbe either be intended or unintended to cause a model to perform poorly. No matter it is anintentional or unintentional adversarial attack, evaluating adversarial examples has become atrend of building a robust deep learning model and understanding the shortcomings of models.First examples of adversarial attacks came from computer vision field:  if we change somepixels in the tensor, that represents the picture, the model classies this image wrongly butfrom human perspective not much changed. In NLP adversarial attacks are more various:

* Changes are perceivable. For human it is easier to notice any changes in words ratherin images (pixels).

* Targeted and untargeted attacks. These are attacks either causing model to classifytext to the specific incorrect class or to any class but the correct one, respectively.

* Semantic or Semantic-lessAttacks can either preserve or change the semantics of thetext.

* White-box and Black-box attacks.White-box attack requires access model infor-mation including network architecture, parameters, inputs, outputs, and other modelattributes. This approach is very effective as it can access anything. In case of black-boxattack we do not have access to attacked model information except the inputs that wefeed in and outputs we get. In our project black-box scenario is considered.

## **2  Related paper description**

The paper "Differentiable Language Model Adversarial Attacks on Categorical SequenceClassifiers" (Fursov et al., 2020) is the main paper related to our project. Authors introduce anapproach to generate adversarial examples based on fine-tuning Masked Language Modellingcalled DILMA (DIfferentiable Language Model Attack). The architecture of the model can beseen at Figure 2. Few words about its constituents.

Language model (denoted as  _Generator: Transformer_ on Figure 2) is a transformer sequence2sequencemasked language model based on BERT. It is pretrained in BERT-style (masked languagemodelling). Sampler  (denoted  as _Straight-Through  Gumbel  Estimator_)  samples  sequences

![NLP adversarial attack](https://github.com/rodrigorivera/mds20_adversarial/blob/main/Project_status_update/images/NLP%20adversarial%20attack.png 'Fig 2. NLP adversarial attack')

_Fig 2. NLP adversarial attack_

 ![input_in_link_formula](https://latex.codecogs.com/svg.latex?x%27=\{x%27_1,%20\dots,%20x%27_n\}$%20from%20logits%20$\{p_1,%20\dots,%20p_n\}) obtained from the language model. Surrogate classifier consists of two pretrained NLP models: CNN text classifier (target model) and bidirectional GRU model. Both models are used for adversarial attack success evaluation.

Let’s briefly describe how it is working:

1. Pretrained _Masked Language Model_(MLM) provides a conditional distribution 𝑝𝜃(𝑥' ̈|𝑥) for given input sequence𝑥, where𝑥is original sequence of tokens and𝑥 ̈is adversarialsequence.

2. Given a conditional distribution we can sample sequences from it ![](https://latex.codecogs.com/svg.latex?x%27%20\sim%20p_{\theta}(x%27|x)). So doesthe sampler(_denoted as ST Gumbel Estimator_).

3. At this stage dierentiable loss function components are calculated:

     Given adversarial sequence we feed it into _Deep Levenstein_ model to measure dif-ference between the original sequence𝑥and the adversarial one 𝑥'. From this part we obtain ![](https://latex.codecogs.com/svg.latex?DL(x%27,x)).

Adversarial sequence is also fed into a substitute classiffier (denoted as _SurrogateClassiffier_). Since we are working with a black-box scenario, no access to the targetedmodel provided. Therefore, substitute classiffier’s scores are used (we assume that ![](https://latex.codecogs.com/svg.latex?C_{y}(x%27)%20\approx%20C^{true}_{y}(x%27)), where ![](https://latex.codecogs.com/svg.latex?C^{true}_{y}(x%27)) is targeted (original) classiffier prediction on 𝑥' and ![](https://latex.codecogs.com/svg.latex?C_{y}(x%27)) is substitute classiffier’s prediction).  Ideally, we want our classiffier make the following ![](C\left(\mathbf{x}_{i}\right) \neq C\left(\mathbf{x}_{i}^{\prime}\right)), in that case an adversarial attack is considered successful.

4. Finally, we obtain the loss function provided below. Notice that this function is differen-tiable, therefore, model BPTT is possible. 

![](https://latex.codecogs.com/svg.latex?\begin{equation}%20%20%20%20L\left(\mathbf{x}^{\prime},%20\mathbf{x},%20y\right)=\beta\left(1-D%20L\left(\mathbf{x}^{\prime},%20\mathbf{x}\right)\right)^{2}-\log%20\left(1-C_{y}\left(\mathbf{x}^{\prime}\right)\right)%20%20%20%20\end{equation})

