---
layout: post
title: "Feature-based Approach with BERT"
author: "Trishala Neeraj"
---

BERT is a language representation model pre-trained on a very large amount of unlabeled text corpus over different pre-training tasks. It was proposed in the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018). BERT is deeply bidirectional, i.e., it pre-trains deep bidirectional representations from text by jointly conditioning on context in both directions. In other words, to represent a word in a sentence, BERT will use both its previous as well as its next context in contrast to: 
1. context-free models like [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (Mikolov et al., 2013) or [Glove](https://nlp.stanford.edu/pubs/glove.pdf) (Pennington et al., 2014)
2. shallowly bidirectional contextual models like [ELMo](https://arxiv.org/abs/1802.05365) (Peters et al., 2018)
3. unidirectional contexual models like [OpenAI GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)(Radford et al., 2018). 

BERT can be fine-tuned on a variety of downstream NLP tasks (such as entailment, classification, tagging, paraphrasing and question answering) without a lot of task-specific modifications to the architecture. Compared to pre-training, fine-tuning is relatively inexpensive and is able to achieve state-of-the-art performance on various sentence-level as well as token-level tasks. The authors have released the [open-source TensorFlow](https://github.com/google-research/bert) implementation.

The paper by Devlin et al. also discusses the advantages of a feature-based approach to directly utilizing the features extracted from the pre-trained model. The authors compare the two approaches (fine-tuning vs feature-based) by applying BERT to the [CoNLL-2003 Named Entity Recognition (NER) task](https://arxiv.org/abs/cs/0306050) (Tjong Kim Sang and De Meulder, 2003). The feature-based approach here comprised of extracting the activations (or contextual embeddings or token representations or features) from one or more of the 12 layers without fine-tuning any parameters of BERT. These embeddings are then used as input to a BiLSTM followed by the classification layer for NER. The authors report that when they concatenate the token representations from the top four hidden layers of the pre-trained Transformer and use that directly in the downstream task, the performance achieved is comparable to fine-tuning the entire model (including the parameters of BERT).

I am fascinated by this result and sought to replicate an experimental setup similar to this work. In this blog post, I will extract and study contextual embeddings from the first sub-token (AKA [CLS] token) at each layer. I will detail my work on a text classification task using a similar feature-based approach that achieves comparable results when utilizing embeddings from any of the 12 Transformer layers in BERT.

# Data

In this blog post, I will work through a text classification task. The dataset I've chosen is the [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification) dataset. The training data is English-only and comprises of the text of comments made in online conversations as well as a boolean field specifying if a given comment has been classified as toxic or non-toxic. The task is to predict if a comment is toxic. Here's a sample of what the data looks like -

| sentence        | label           | 
| ------------- |:-------------:| 
| Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info.      | 0 | 
| Worse case scenario we do add it with overview (make it an extended history/misison/overview) Banner has been added to talkpage and subheadings to the SandBox page start posting info and change headings if needed but be sure to indicate so in the edit summary bar      | 0      |

I'd be working with a small sample of 1000 comments to get started.

# Tokenization

BERT’s model architecture is a multi-layer bidirectional Transformer encoder. In this blog, I'd be working with the BERT "base" model which has 12 Transformer blocks or layers, 16 self-attention heads, hidden size of 768.

Input data needs to be prepared in a special way. BERT uses WordPiece embeddings (Wu et al.,2016) with a 30,000 token vocabulary. There are 2 special tokens that are introduced in the text -- 

* a token [SEP] to separate two senteces, and 
* a classification token [CLS] which is the first token of every tokenized sequence.

The authors state that the final hidden state corresponding to the [CLS] token is used as the aggregate sequence representation for classification tasks. I am interested in looking into how meaningful these representations are across each of the 12 layers. Here's how to obtain these embeddings from the dataset I've selected.

### Tokenize each sentence, i.e., split each sentence into tokens

For example, consider the following non-toxic comment from the dataset:
```
Please stop vandalising wikipedia immediately or you will be blocked
```

This sentence upon tokenization would look like this:
```
'please', 'stop', 'van', '##dal', '##ising', 'wikipedia', 'immediately', 'or', 'you', 'will', 'be', 'blocked'
```

Notice, the tokens are either the words lowercases, or a leading-subword from a larger word or a trailing-subword from a larger word (marked with a `##` to indicate so).

### Add special tokens -- [CLS] in the beginning & [SEP] as a separator between sentences (or at the end of a single sentence)

```
[CLS], 'please', 'stop', 'van', '##dal', '##ising', 'wikipedia', 'immediately', 'or', 'you', 'will', 'be', 'blocked', [SEP]
```

### Pad tokenized sequences to the maximum length (or truncate sequences to a fixed size)

For this dataset, I've chosen the maximum length of 64. Sequences with lesser than 64 tokens will be padded to meet this length, and sequences with more would be truncated.

### Convert tokens to IDs and convert to tensors

```
tensor([  101,  3531,  2644,  3158,  9305,  9355, 16948,  3202,  2030,  2017,
         2097,  2022,  8534,   102,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0])
```

### Create an attention masks (also tensors) to explicitly identify tokens that are actually PAD tokens.

```
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

I utilized the `BertTokenizer` made available as part of the [Hugging Face Transformer library](https://huggingface.co/transformers/) to perform the above operations. The code snippet below gets us the following 2 tensors to feed into the model later - 
* input_ids
* attention_masks

They should both have the same shape of (num_samples x max_len), in the case of this work -- (1000 x 64).

```
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []
for sentence in df['sentence'].tolist():
    dictionary = tokenizer.encode_plus(
                        sentence,                      
                        add_special_tokens = True,
                        max_length = 64,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    # encode_plus returns a dictionary 
    input_ids.append(dictionary['input_ids'])
    attention_masks.append(dictionary['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
```


# Contextual Embeddings

At this point, I have the input IDs and attention masks for the text data, and the next step is to extract and study the embeddings. For this, I used the `bert-base-uncased` pre-trained model made available by Hugging Face transformers. As I mentioned in the above section, it is trained on lower-cased English (12-layer, 768-hidden, 12-heads, 110M parameters). When loading the pre-trained model, make sure to set `output_hidden_states` parameter True which will give us access to output embeddings from all 12 Transformer layers. 

The snippet below shows that the model takes as input `input_ids` -- the indices obtained in the previous section, as well as attention masks.

```
config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
model = BertModel.from_pretrained("bert-base-uncased", config=config)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)
```

The tuple `outputs` comprises of the following tensors(based on [documentation](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel) and [this GitHub Issue](https://github.com/huggingface/transformers/issues/1827)):
* last_hidden_state
* pooler_output
* hidden_states

In this work, I'm most interested in the `hidden_states` which is a tuple of 3 tensors. The last element of this tuple contains the contextual embeddings that each Transformer layer outputs. They can be accessed like this -

```
embeddings = outputs[2][1:]
```

Here, `embeddings` is another tuple of size 12, containing 12 sets of contextual embeddings -- one from each layer. Note that these embeddings represent all tokens in the input sequence padded / truncated to the maximum fixed length. Each tensor in `embeddings` is of the shape -- (num_samples x max_length x hidden_size), in this case (1000 x 64 x 768).

In this work, I'm focussed on the downstream task of text classification, and therefore would only work with representations encoding [CLS] tokens, because as mentioned in the previous section, they are the aggregate sequence representation for classification tasks. Therefore, I will slice the embeddings -

```
def get_CLS_embedding(layer):
    return layer[:, 0, :].numpy()
    
cls_embeddings = []
for i in range(12):
    cls_embeddings.append(get_CLS_embedding(embeddings[i]))
```
Each of these 12 NumPy arrays in the list `cls_embeddings` would be of shape (num_samples x hidden_size), in this case (1000 x 768), and each of these can be used as train a text classifier. As a reminder, I will not be fine-tuning, but simply using these features extracted from BERT as input features to my model.

# Performance on Text Classification

I trained a Logistic Regression model using these `cls_embeddings` as features, one layer at a time. Below is the accuracy for each layer on a fixed validation set.


| cls_embedding from layer #        | accuracy           | 
| ------------- |:-------------:| 
| 1      | 0.812 | 
| 2      | 0.86      | 
| 3      | 0.856 | 
| 4      | 0.868      | 
| 5      | 0.868 | 
| 6      | 0.844      | 
| 7      | 0.848 | 
| 8      | 0.804      | 
| 9      | 0.824 | 
| 10      | 0.844      | 
| 11      | 0.828      | 
| 12      | 0.856      | 

From the results, it looks like more than one of the 12 layers helps achieve good accuracies without any fine-tuning, and using a basic logistic regression model.

# Future Work

In future work, I will also fine-tune rather than sticking with the feature-based approach. While the latter might be a good and economical option for a large number of use-cases, it would be definitely worth exploring tradeoffs to using it over fine-tuning. 