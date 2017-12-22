---
layout: post
title: "Semantic Entailment"
author: "Trishala Neeraj"
---

A key part of our understanding of natural language is the ability to understand sentence semantics. <!-- Essentially, given a piece of text the goal is to understand the additional information that the reader is entitled to conclude about what the author believes in addition to what she actually says.  -->

Semantic Enatilment or, more poularly, the task of Natural Language Inference (NLI) is a core Natural Language Understanding task (NLU). While it poses as a classification task, it is uniquely well-positioned to serve as a benchmark task for research on NLU. It attempts to judge whether one sentence can be inferred from another. 

More specifically, it tries to identify the relationship between the meanings of a pair of sentences, called the premise and the hypothesis. The relationship could be one of the following:

* Entailment: the hypothesis is a sentence with a similar meaning as the premise
* Contradiction: the hypothesis is a sentence with a contradictory meaning
* Neutral: the hypothesis is a sentence with mostly the same lexical items as the premise but a different meaning.

Consider the following example taken from the RTE 1 suite:
>T: The country's largest private employer, Wal-Mart Stores Inc., is being sued by a number of its female employees who claim they were kept out of jobs in management because they are women.
>H: Wal-Mart sued for sexual discrimination.

Classifying into one of entailment, contradiction and neutral requires understanding that keep someone out of a job is here a another way of saying that someone was not hired, and the knowledge that not hiring someone for a job because of the applicant’s gender is an act of sexual discrimination, which is against the law. Succeeding at NLI requires a model to fully capture sentence meaning by handling complex linguistic phenomena like lexical entailment, quantification, coreference, tense, belief, modality, and ambiguity - both lexical and syntactic. NLI is is essential in tasks such as information retrieval, semantic parsing and commonsense reasoning, to name a few.

My first blog post surveys the evolution of semantic entailment over the last twenty years. This task has been of interest to linguists and computational linguits for decades and I hope to describe the several variations in the task, the proposed approaches to solve those task, as well as the current state of the art.

Some variations of the task that I plan to discuss:

* Framework for Computational Semantics 1996 [FraCaS]
* Recognizing Textual Entailments Challenges 2005-2011 [RTE]
* Sentences Involving Compositional Knowledge Dataset 2015 [SICK]
* Stanford Natural Language Inference Dataset [SNLI]
* Multi-Genre Natural Language Inference Dataset 2017 [MultiNLI]

# Data

Below is a discussion of the main contributions of and biases in each of these datasets that shaped the NLI task over the years followed by examples tabulated. 

## FraCaS

The [FraCas](https://pdfs.semanticscholar.org/6bdc/820d01c14ac866e1db85e105947f703c596b.pdf) project was can be seen as the starting point of modern approach to textual inference. It was a consortium sponsored by the European Union in 1996, a huge effort with the aim to develop a range of resources related to computational semantics. The main contribution of this project was a set of 346 inference problems in addition to the fact that they framed Natural Language Inference Tasks as a 3 way classification problem: Yes/ True, No/ False and Undefined/ UNK, mapping to entails, contradicts and neither respectively. The suite <!-- is structured according to the semantic phenomena involved in the inference process for each example, and  -->contains 9 sections including quantifiers, adjectives, comparatives, plurals etc. 

The FraCaS testsuites were intended to be for semantics, analogous to syntactic testsets that were created for evaluating grammar. However, the project was not well tested, documented or followed-up.

Another aspect of FraCas that was rather controversial, <!-- is that it that the semantic relations it assumes between premises and hypotheses  --> is that it assumes the sematic relations between premise and hypothesis are only based on the semantics of the particular construction and the lexical meaning of the words involved. Ideally, the dataset should contain examples where the label would depend on using the premise in addition to some context, i.e., knowledge about the world. 

Despite its obvious cons, i.e., small size of the dataset and artificial distribution (examples were lab-made), it seems to provide a comprehensive coverage of semantic phenomena, and hence, a well regarded testsuite to test logical approaches as regards NLI. FraCaS data was used to evaluate [NatLog](https://nlp.stanford.edu/projects/natlog.shtml), which aims to tackle entailment problems based on monotonicity of generalized quantifiers.


## RTE Challenges

[Recognizing Textual Entailment (RTE)](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) challenge tasks were at some point the primary sources of annotated NLI corpora. These are generally high-quality, hand-labeled data sets, and they have stimulated innovative logical and statistical models of natural language reasoning. They are however, very small in size and hence not ideal for testing learned distributed representations. 

RTE challenges were launched to become a task organized year after year with steady improvements in the development process and increase in difficulty. All RTE data sets looked at the enatailment problem as a sub-task of NLP real-life applications (eg. Question Answering). Thus, all of them conatined more natural examples compared to FraCas.

However, RTE evaluations were not testing for logical entailment but a less strict relation. For this very reason people started preferring the more general term Natural Language Inference over Recognizing Textual Entailment.


## SICK

[SICK](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.5834&rep=rep1&type=pdf) was created for a shared task in SemEval - 2014. It has about 10,000 premise-hypothesis examples annotated for similarity and the semantic relation (entailment, contradiction, neither ). The sick examples were derived from descriptions of images and video snippets created by Turkers. More specifically, the 8K ImageFlickr dataset and the SemEval-2012 STS MSR-Video Descriptions dataset were used. From each seed sentence up to three new sentences were created manually: a sentence with a similar meaning, a sentence with a contradictory meaning, and lastly, a sentence with mostly the same lexical items but a different meaning. The seed examples were captions provided by Turkers for images, while the extensions were created by people who developed the dataset - linguists.

SICK was an easier dataset compared to RTE since its purpose was to create a compositional semantics suite that did not require named-entity recognition or encyclopedic knowledge about the world. The semantic relation between premise and hypothesis was meant to be decidable on purely linguistic evidence.

This method of creating the dataset - deriving the sentences from image captions introduces biases. In these datasets contradiction is not semantic contradiction. What contradiction here means is simply that premise and hypothesis are not captions for the same picture, and obviously, that does not necessarily imply a contradiction. 

## SNLI

SNLI was introduced in 2015 after recognizing that machine learning research in this area of semantic enatilment has been very limited by the lack of large-scale resources. SNLI is significantly larger than any other corpus for NLI. It offers 570K pairs of labeled sentences written in a grounded and naturalistic context, based on image captioning. The increase in scale allowed neural network-based models to perform competitively on NLI benchmarks for the first time. NLI is also meant to be used as a tool for evaluating domain-general approaches to semantic representation, in addition to be used for training modern models that require enormous amounts of data.

The authors also sought interannotator agreement which was completely absent from the development of other exisiting corpora. However, since the method of putting together this dataset was very similar to SICK, only difference being that in SNLI, extension steps were outsourced to Turkers, bias due to being image-caption based was also present in SNLI.


## MultiNLI

MultiNLI corpus was designed in 2017 for use in the development and evaluation of machine learning models for sentence understanding. It was presented in RepEval - 2017 and is the most newest and by far the largest corpora available for NLI tasks and it improves upon available resources in its coverage. As will be seen in the table below, the examples from the dataset all contain a pair of sentences and the judgement of five turkers and a consensus judgement.

The premises in MultiNLI are not rephrased from image captions, unlike SNLI and SICK. Hence, it is not as skewed as them and negative premises are not exceptionally rare. Moreover, it offers data from ten distinct genres of written and spoken English, creating a great setting for evaluation of cross-genre domain adaptation which is often a hard task. The fact that it is multi-genre ensures that any system is being evaluated on nearly the full complexity of the language. 

<!-- Consequently, large datasets, coupled with a much easier version of the task created the perfect playground for experimentation with Deep Learning models. -->

| Corpus | Sentence 1 | Sentence 2 | Relationship |
| ------ | ------ | ------ | ------ |
| MultiNLI | Met my first girlfriend that way. | I didn’t meet my first girlfriend until later. | FACE-TO-FACE contradiction C C N C |
| | He turned and saw Jon sleeping in his half-tent. | He saw Jon was asleep. | FICTION entailment N E N N |
| | 8 million in relief in the form of emergency housing. | The 8 million dollars for emergency housingwas still not enough to solve the problem. | GOVERNMENT neutral N N N N |
| SNLI | A man inspects the uniform of a figure in some East Asian country. | The man is sleeping. | contradiction C C C C C |
| | An older and younger man smiling. | Two men are smiling and laughing at the cats playingon the floor. | neutral N N E N N | 
|  | A black race car starts up in front of a crowd of people. | A man is driving down a lonely road. | contradiction C C C C C |
| SICK | Two teams are competing in a football match. | Two groups of people are playing football. | entailment |
| | The brown horse is near a red barrel at the rodeo. | The brown horse is far from a red barrel at the rodeo.  | contradiction |
| | A man in a black jacket is doing tricks on a motorbike. | A person is riding the bicycle on one wheel. | neutral |
| RTE | The Republic of Yemen is an Arab, Islamic and independent sovereign state whose integrity is inviolable, and no part of which may be ceded. | The national language of Yemen is Arabic. | True |
| | Most Americans are familiar with the Food Guide Pyramid– but a lot of people don’t understand how to use it and the government claims that the proof is that two out of three Americans are fat. | Two out of three Americans are fat. | True |
| | Regan attended a ceremony in Washington to commemorate the landings in Normandy. | Washington is located inNormandy. | False |
| FraCaS | An Irishman won the Nobel prize for literature. | An Irishman won a Nobel prize. | Did an Irishman win a Nobel prize? [Yes, FraCaS 017] |
| | No delegate finished the report. | No delegate finished the report on time. | Did any delegate finished the report on time? [No, FraCaS 038] |
| | Smith, Jones or Anderson signed the contract. | Jones signed the contract. | Did Jones sign the contract? [UNK, FraCaS 083] |

# Architectures

The semantic entailment architectures have varied over the years. NLI has been addressed using a variety of techniques, including those based on symbolic logic, knowledge bases, and neural networks. 

The [Hickl-Bensley System Architecture](http://delivery.acm.org/10.1145/1660000/1654571/p171-hickl.pdf?ip=69.201.189.30&id=1654571&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&CFID=1018554410&CFTOKEN=42843982&__acm__=1513809516_a9b9082538ff87997ea26e7702c95729) was one of the earliest semantic entailment architectures. The architecture diagram is shown below.

![](https://drive.google.com/uc?export=&id=1wdIoSJ4mbgw33_vIQLDrd75BQyMMvaTq "Generic")

The sentence pairs are fed into a preprocessing module and the annotated passages are then sent to a Commitment
Extraction module. Each pair of commitments are then considered in turn by an Entailment Classification module. 
If a commitment pair is judged to be a positive instance of TE, it is sent to an Entailment Validation module. If no text commitment can be identified which contradicts the hypothesis, it is presumed to be textually entailed (return YES). If the entailed hypothesis is textually contradicted by any of the commitments extracted from the premise, the hypothesis is considered to be contradicted by the premise, NO is returned.

From traditional modules like the one discussed above, to newer deep learning architectures, the task of semantic entailment has evolved significantly.

To study the current state-of-the-art approaches for this task, I looked at the results of the results of the [Shared Task at RepEval 2017](https://repeval2017.github.io/shared/). The Shared Task evaluated neural network sentence representation learning models on the MultiNLI corpus I described above. There were 5 participating teams and they all beat the baselines of BiLSTM and CBOW reported in Williams et al. 


<!-- INSERT DETAILS ABOUT THE COMPETITION
The best single model used stacked BiLSTMs with residual connections to extract sentence features and reached 74.5% accuracy on the genre-matched test set. Surprisingly, the results of the competition were fairly consistent across the genrematched and genre-mismatched test sets, and across subsets of the test data representing a variety of linguistic phenomena, suggesting that all of the submitted systems learned reasonably domainindependent representations for sentence meaning.
 -->

The Deep learning architecture often is a variant of the following architecture as shown in the figure below.

![](https://drive.google.com/uc?export=&id=1yNyFuZDZkVD8ki4-vosikPyowyXOAEon "Generic")

Models for semantic entailment task generally fall into two categories: 1) Sentence encoding of the individual sentences 2) Joint methods that allow to use encoding of both sentences, like cross-features of attention. Approach 1, is useful for training generic sentence encoders and can be used for a variety of other tasks, like the [InferSent](https://arxiv.org/pdf/1705.02364.pdf) paper.
The figure above shows a generic architecture, for SNLI tasks. The sentence encoders are often a deep learning architecture, like a CNN or an RNN that outputs a fixed size representation vector v, for the hypothesis and the premise separately. Once
the sentence vectors are generated, 3 matching
methods are applied to extract relations between the two fixed size representation vectors.
(i) concatenation of the two representations
(ii) element-wise product of the two vectors; and
(iii) absolute element-wise difference of the two vectors. The
resulting vector, which captures information from
both the premise and the hypothesis, is fed into
a 3-class classifier consisting of multiple fullyconnected
layers culminating in a softmax layer. 

The generic approach is best highlighted in a[paper by Nie and Bansal](https://arxiv.org/pdf/1708.02312.pdf) which is also one of the state of the art architectures for semantic entailment as per the results on the RepEval 2017 results. 

![](https://drive.google.com/uc?export=&id=1KpPTdhWDE3M2ZHYXAaRKJMNk_hCzi2aT "Nie and Bansal")

Their architecture relies on word vectors and Bi-LSTMs connected in a ResNet-like architecture. Each of the subsequent layer concatenates the vector representation of the previous layers. Using the vector representation from every previous layer, means the architecture, becomes more computationally expensive, but such architectures have performed well for image classification tasks in the past. The concatenation of all the vectors in the last layer passes through a row max pooling which creates a final vector representation,that passes through a 3-way softmax for the 3 categories of entailment, contradiction, or neural.

[RNN-based sentence encoder with gated attention](https://arxiv.org/abs/1708.01353) is another top result from the same competition.

# Future Directions

Learning generic sentence embeddings have barely been explored and understanding the NLI task can improve sentence encoding. The MultiNLI and the SNLI datasets are large scale labeled datasets for Natural Language Processing. Unlike images, NLP does not have large scale labeled datasets that can be used for equivalent transfer learning tasks. The infersent paper, showed how transfer learning is a viable approach for learning different tasks and how such models perform competitively for tasks like Sentiment Analysis  and text classification. Another paper, [StarSpace](https://arxiv.org/abs/1709.03856), shows embedding sentences, words for various tasks like text classification perform very competitively yet extensive comparison of sentence encoding architectures with NLI has not yet been done. More work need to done, to better understand sentence encoding architectures and I believe semantic entailment datasets will play a key role in that line of research.