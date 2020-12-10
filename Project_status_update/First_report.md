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


![NLP adversarial attack](https://github.com/rodrigorivera/mds20_adversarial/blob/main/Project_status_update/images/NLP%20adversarial%20attack.png 'Fig 1. NLP adversarial attack')

_Fig 1. NLP adversarial attack_


 ![input_in_link_formula](https://latex.codecogs.com/svg.latex?x%27=\{x%27_1,%20\dots,%20x%27_n\}$%20from%20logits%20$\{p_1,%20\dots,%20p_n\}) obtained from the language model. Surrogate classifier consists of two pretrained NLP models: CNN text classifier (target model) and bidirectional GRU model. Both models are used for adversarial attack success evaluation.

Let’s briefly describe how it is working:

1. Pretrained _Masked Language Model_(MLM) provides a conditional distribution 𝑝𝜃(𝑥' ̈|𝑥) for given input sequence𝑥, where𝑥is original sequence of tokens and𝑥 ̈is adversarialsequence.

2. Given a conditional distribution we can sample sequences from it ![](https://latex.codecogs.com/svg.latex?x%27%20\sim%20p_{\theta}(x%27|x)). So doesthe sampler(_denoted as ST Gumbel Estimator_).

3. At this stage differentiable loss function components are calculated:

     Given adversarial sequence we feed it into _Deep Levenstein_ model to measure dif-ference between the original sequence𝑥and the adversarial one 𝑥'. From this part we obtain ![](https://latex.codecogs.com/svg.latex?DL(x%27,x)).

Adversarial sequence is also fed into a substitute classiffier (denoted as _SurrogateClassiffier_). Since we are working with a black-box scenario, no access to the targetedmodel provided. Therefore, substitute classiffier’s scores are used (we assume that ![](https://latex.codecogs.com/svg.latex?C_{y}(x%27)%20\approx%20C^{true}_{y}(x%27)), where ![](https://latex.codecogs.com/svg.latex?C^{true}_{y}(x%27)) is targeted (original) classiffier prediction on 𝑥' and ![](https://latex.codecogs.com/svg.latex?C_{y}(x%27)) is substitute classiffier’s prediction).  Ideally, we want our classiffier make the following ![](https://latex.codecogs.com/svg.latex?C\left(\mathbf{x}{i}\right)%20\neq%20C\left(\mathbf{x}{i}^{\prime}\right)), in that case an adversarial attack is considered successful.

4. Finally, we obtain the loss function provided below. Notice that this function is differen-tiable, therefore, model BPTT is possible. 

![](https://latex.codecogs.com/svg.latex?%20L\left(\mathbf{x}^{\prime},%20\mathbf{x},%20y\right)=\beta\left(1-D%20L\left(\mathbf{x}^{\prime},%20\mathbf{x}\right)\right)^{2}-\log%20\left(1-C_{y}\left(\mathbf{x}^{\prime}\right)\right))   [1]

5. Update LM’s weights according to the loss function (update ![](https://latex.codecogs.com/svg.latex?\theta_{i-1}) using gradient descentand get new weights ![](https://latex.codecogs.com/svg.latex?\theta_{i})) via backward pass value.  Note that this updating processtake into account two terms: maximising the probability drop and minimising the editdistance. So it should be as close to 1 as possible.

Also, for more DILMA model details one can take a look at Figure 3, where the process describedabove is shown more precisely.

![NLP adversarial attack](https://github.com/rodrigorivera/mds20_adversarial/blob/main/Project_status_update/images/Dilma_model_arch.png 'Fig 2. DILMA model architecture')

_Fig 2. DILMA model architecture_


## **3 Project related challenges and tasks**

**DILMA model nuance:** each update procedure for each x starts from the pretrained LM parameters ![](https://latex.codecogs.com/svg.latex?\theta_{0}). An adversarial attack is based on a masked language model (MLM). DILMAuse fine tuned parameters of MLM by optimising a weighted sum of two differentiable termsbased on a surrogate distance between sequences and a surrogate classifier model scores. So,the main challenge is to retrain the model to generate adversarial examples without localadaptation to each sequence.

**Implementation:** we need to reimplement model using PyTorch deep learning framework.

**Discrete sequences are more challenging than Pictures:** There are two main challengesfor adversarial attacks on discrete sequence models: a discrete space of possible objects and acomplex definition of a semantically coherent sequence.

**Finding universal approach for attack:** Our updated approach can be based on generationof adversarial sequences like in paper "Generating Natural Language Adversarial Exampleson a Large Scale with Generative Models" (Yankun Ren et al., 2020).  Model generates textadversarial examples from scratch, adversarial examples are not restricted to existing inputs.Pretrained model can generate an unlimited number of adversarial examples without any inputdata. By the way, model generates adversarial texts without querying the attacked model, thusthe generation procedure became faster. However, they use RNNs seq2seq models and lengthof sequences they can process is limited.

**Investigating existing architecture**:

![NLP adversarial attack](https://github.com/rodrigorivera/mds20_adversarial/blob/main/Project_status_update/images/Dilma_arch_impl.png 'Fig 3. DILMA model architecture implementation')

_Fig 3. DILMA model implementation_


## **4 Contributions**

### **Daniil Moskovskiy**

Literature review, model architecture implementation on PyTorch, main model training, taskdistribution for teammates, running the proposed DILMA attack for at one NLP dataset (TRECdataset).

### **Margarita Sharkova**

Literature review, model architecture implementation on PyTorch, classifiers (target and substitute) training, report preparation, retraining the model to generateadversarial examples without local adaptation to each sequence.

### **Aleksandr Esin**

Literature review, model result evaluation (attack success rate, perplexity, manual processingof generated texts), building a graphics for the model and results visualization; report andpresentation preparation.

## **5 Links**

The implementation will be available at our team’s github repository.

Referenced papers:  : [Differentiable Language Model Adversarial Attacks on Categorical Se-quence Classifers ](https://arxiv.org/pdf/2006.11078.pdf), [PyTorch framework](https://pytorch.org/).

### References

1. I. Fursov and A. Zaytsev and N. Kluchnikov and A. Kravchenko and E. Burnaev, DifferentiableLanguage Model Adversarial Attacks on Categorical Sequence Classifier, 2020

2. Yankun Ren, Jianbin Lin, Siliang Tang, Jun Zhou, Shuang Yang, Yuan Qi, and Xiang Ren.Generating natural language adversarial examples on a large scale with generative models,2020.



