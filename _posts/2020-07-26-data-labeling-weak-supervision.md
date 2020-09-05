---
layout: post
title: "Data Labeling using Weak Supervision: In Action"
author: "Trishala Neeraj"
---

In this blog post, I will share my takeaways and results from using Weak Supervision to label Jigsaw's Comments [data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) as toxic or non-toxic comments. In my previous [blog post](https://trishalaneeraj.github.io/2020-07-05/weak-supervision), I've discussed in detail how human annotations are not only expensive but also not always straightforward to reliably make use of in training machine learning applications. Moreover, annotators may be having to [repeatedly](https://twitter.com/kirbyconrod/status/1139184438846402562) read and analyze comments targeted towards the underrepresented group they identify with which can take a toll on their well-being. Give it a read to learn more about why this problem motivates me and why I seek to explore Weak Supervision for labeling this data at scale.

In this blog post, I will detail the end-to-end workflow - 

![Image with caption](https://drive.google.com/uc?export=view&id=19F4m908Hu_R22XTeI9J2iIZqHLSyTAl9 "Overview")

I used 90% of the Jigsaw English train set as my train set. Note that I will not be using the ground truth for training as the idea is to programmatically label these data points using Weak Supervision, and therefore, would treat this as an unlabeled dataset. For the remaining 10%, I will use the labels -- 5% for the validation set and 5% makes the sequestered test set that I will only test on once at the very end. My train set has a little over 200K comments.

I used [Snorkel](https://github.com/snorkel-team/snorkel), an open-source library for Weak Supervision by Hazy Research Lab at Stanford for this project. 

# Developing Labeling Functions
In my previous [blog post](https://trishalaneeraj.github.io/2020-07-05/weak-supervision), I described Labeling Functions (LFs) in detail and also shared my initial thoughts on what they would look like for the problem of labeling for toxicity.

Here I describe some LFs that capture heuristics a human annotator might use in determining if a comment is toxic or not. Each LF would take a text input (comment), and based on the logic defined, it would return a label (toxic or non-toxic), or simply not label. I experimented with various kinds of LFs -

## Pre-trained Models
If a comment mentioned titles of books, songs, or other pieces of known art, I labeled it as non-toxic as it’s a signal of it being specific to a conversation topic and less likely to be toxic. With a similar intuition, if a comment contains at least 3 mentions of named entities, I label it as non-toxic. For both of these rules, I use SpaCy’s pre-trained Named Entity Recognition models.

{% highlight python %}from snorkel.preprocess.nlp import SpacyPreprocessor
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

@labeling_function(pre=[spacy])
def contains_work_of_art(x):
    """If comment contains titles of books, songs, etc., label non-toxic, else abstain"""
    if any([ent.label_ == "WORK_OF_ART" for ent in x.doc.ents]):
        return NONTOXIC
    else:
        return ABSTAIN
    
@labeling_function(pre=[spacy])
def contains_entity(x):
    """If comment contains least 3 mentions of an entity, label non-toxic, else abstain"""
    if len([ent.label_ in ["PERSON", "GPE", "LOC", "ORG", "LAW", "LANGUAGE"] for ent in x.doc.ents])>2:
        return NONTOXIC
    else:
        return ABSTAIN
{% endhighlight %}

In another pair of LFs, I use [TextBlob](https://textblob.readthedocs.io/en/dev/)’s pre-trained sentiment analysis model. The intuition here is that non-toxic comments are likely to have higher polarity (+1 is positive, -1 is negative) and high subjectivity scores.

{% highlight python %}from snorkel.preprocess import preprocessor
from textblob import TextBlob
@preprocessor(memoize=True)

def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x

@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    """If comment has a polarity score between +0.9 and +1, label non-toxic, else abstain"""
    return NONTOXIC if x.polarity > 0.9 else ABSTAIN

@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x):
    """If comment has a subjectivity score between +0.7 and +1, label non-toxic, else abstain"""
    return NONTOXIC if x.subjectivity >= 0.7 else ABSTAIN
{% endhighlight %}

I used an open-source library, [better_profanity](https://pypi.org/project/better-profanity/) to also detect swear words and their various [leetspeak](https://en.wikipedia.org/wiki/Leet) versions in the comments. Intuitively, it is likely to make someone uncomfortable and leave the online discussion (which is how Jigsaw defines toxicity).

{% highlight python %}from better_profanity import profanity
@labeling_function()
def contains_profanity(x):
    """
    If comment contains profanity label toxic, else abstain. 
    Profanity determined using this library - https://github.com/snguyenthanh/better_profanity
    """
    return TOXIC if profanity.contains_profanity(x.text) else ABSTAIN
{% endhighlight %}

## Pattern Matching
I observed a few commonly occurring phrases, particularly in the non-toxic comments -
* “Please read this...” - usually comments linking readers to guidelines of various forums, or other useful resources.
* “Please stop vandalizing...” - usually comments directed at those making toxic comments or causing other issues.
* “Please don’t harass me...” - usually comments directed at those making targeted toxic comments.
* “I will report you...” - usually comments directed at those making toxic comments or causing other issues.

I wrote LFs that can match many variations of these phrases using SpaCy. I've included examples in docstrings - 

{% highlight python %}from snorkel.preprocess.nlp import SpacyPreprocessor
spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

@labeling_function(pre=[spacy])
def contains_pleaseread(x):
    """
    Will match commonly occuring phrases like - 
    Please read this
    Please read the
    Please read
    """
    matcher = Matcher(nlp.vocab)
    pattern = [{"LEMMA": "please"},
               {"LEMMA": "read"},
               {"LEMMA": "the", "OP": "?"},
               {"LEMMA": "this", "OP": "?"}]
    matcher.add("p1", None, pattern)
    matches = matcher(x.doc)
    return NONTOXIC if len(matches)>0 else ABSTAIN

@labeling_function(pre=[spacy])
def contains_stopvandalizing(x):
    """
    Will match commonly occuring phrases like - 
    stop vandalizing
    do not vandalize
    don't vandalize
    """
    matcher = Matcher(nlp.vocab)
    pattern1 = [{"LEMMA": "do"},
                {"LEMMA": "not"},
                {"LEMMA": "vandalize"}]
    pattern2 = [{"LEMMA": "stop"}, 
                {"LEMMA": "vandalize"}]
    matcher.add("p1", None, pattern1)
    matcher.add("p2", None, pattern2)
    matches = matcher(x.doc)
    return NONTOXIC if len(matches)>0 else ABSTAIN
    
@labeling_function(pre=[spacy])
def contains_harassme(x):
    """
    Will match commonly occuring phrases like - 
    harass me
    harassed me
    harassing me
    """
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "harass"}, 
               {"LOWER": "me"}]
    matcher.add("p1", None, pattern)
    matches = matcher(x.doc)
    return NONTOXIC if len(matches)>0 else ABSTAIN

@labeling_function(pre=[spacy])
def contains_willreport(x):
    """Will match commonly observed phrases like - 
    report you
    reported you
    reporting you
    reported your
    """
    matcher = Matcher(nlp.vocab)
    pattern = [{"LEMMA": "report"}, 
               {"LEMMA": "you"}]
    matcher.add("p1", None, pattern)
    matches = matcher(x.doc)
    return NONTOXIC if len(matches)>0 else ABSTAIN
{% endhighlight %}

I looked for URLs and email addresses in comments, and if found, labeled them non-toxic. The intuition here is that they are likely to be informative (asking readers to find more info or reach out for whatever reason). This might also indicate spams (ads or self-promoting business), but overall less likely to offend anyone due to toxicity.

{% highlight python %}@labeling_function(pre=[spacy])
def contains_email(x):
    """If comment contains email address, label non-toxic, else abstain"""
    matcher = Matcher(nlp.vocab)
    pattern = [{"LIKE_EMAIL": True}]
    matcher.add("p1", None, pattern)
    matches = matcher(x.doc)
    return NONTOXIC if len(matches)>0 else ABSTAIN
    
@labeling_function(pre=[spacy])
def contains_url(x):
    """If comment contains url, label non-toxic, else abstain"""
    matcher = Matcher(nlp.vocab)
    pattern = [{"LIKE_URL": True}]
    matcher.add("p1", None, pattern)
    matches = matcher(x.doc)
    return NONTOXIC if len(matches)>0 else ABSTAIN
{% endhighlight %}

## Keyword Searches

I also looked for use of profanity in a comment using an external knowledge base found [here](https://code.google.com/archive/p/badwordslist/downloads). The better_profanity library consumes part of this list so these 2 LFs might be very correlated. Thinking along similar lines, I look for words like “thank you” and “please” (and their variations), and label those as non-toxic, as they are indications of civil conversations (more often than when used sarcastically).

{% highlight python %}def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN

def make_keyword_lf(keywords, label=TOXIC):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

with open('../../../Downloads/public_datasets/badwords.txt') as f:
    toxic_stopwords = f.readlines()

toxic_stopwords = [x.strip() for x in toxic_stopwords] # len = 458
"""Comments mentioning at least one of Google's Toxic Stopwords 
https://code.google.com/archive/p/badwordslist/downloads are likely toxic"""
keyword_toxic_stopwords = make_keyword_lf(keywords=toxic_stopwords)

keyword_pl = make_keyword_lf(keywords=["please", "plz", "pls", "pl"], label=NONTOXIC)

keyword_thanks = make_keyword_lf(keywords=["thanks", "thank you", "thx", "tx"], label=NONTOXIC)
{% endhighlight %}

## Miscellaneous 
I found that many comments written in all caps were often toxic -

{% highlight python %}@labeling_function()
def capslock(x):
    """If comment is written in all caps, label toxic, else abstain"""
    return TOXIC if x.text == x.text.upper() else ABSTAIN
{% endhighlight %}

Next, I applied these LFs to the train set, obtaining the Label Matrix for the train set. It’s a NumPy array with one column for each LF we create and one row for one data point. We need it to train the generative model in the next step. I used Snorkel's [LFAnalysis](https://snorkel.readthedocs.io/en/v0.9.1/packages/_autosummary/labeling/snorkel.labeling.LFAnalysis.html) utility which summarizes the coverage, overlaps, and conflicts between these LFs, and helps get a sense of how these LFs are doing. Since we're assuming that we don't have labels to these comments, we compute accuracies.

As a reminder, LFs are expected to be noisy, conflicting, and potentially correlated. In the table below - 
* Polarity is the set of unique labels this LF outputs (excluding abstains). In this case, I’ve made all my LFs unipolar, i.e., they label only 1 or 0, but never both (although one can choose to do this the other way). 
* Coverage is the fraction of data this LF labels. Higher the better, but that often comes with false positives. The keyword lookup LF for please and its variants labels the maximum amount (38%) of data single-handedly. LFs contains_entity and keyword-based searches of profane words have high coverages as well. On the other hand, all phrase-pattern matching LFs have among the lowest coverages. I will still retain them as they encode useful information and seemed to identify non-toxic comments based on the spot checks I performed. 
* Overlaps column represents the fraction of data that is labeled by this LF and at least one more LF. For example, 28.7% of the data is labeled by the LF keyword_please, and at least one more LF.
* Conflicts column represents the fraction of data for which this and another LF disagree. For example, 14.5% of data is labeled in disagreement by LF keyword_please and other LFs.

|                          | j  | Polarity | Coverage        | Overlaps        | Conflicts       |
|--------------------------|----|----------|-----------------|-----------------|-----------------|
| contains_work_of_art     | 0  | [0]      | 0.08050935913   | 0.07522590137   | 0.04089088144   |
| contains_entity          | 1  | [0]      | 0.345457618     | 0.2762507828    | 0.1517291768    |
| textblob_polarity        | 2  | [0]      | 0.004607493265  | 0.003598516854  | 0.0008996292136 |
| textblob_subjectivity    | 3  | [0]      | 0.1170909669    | 0.07056870483   | 0.03657663747   |
| contains_profanity       | 4  | [1]      | 0.08198057596   | 0.07962961122   | 0.05119436961   |
| contains_pleaseread      | 5  | [0]      | 0.004160163822  | 0.004160163822  | 0.001381750947  |
| contains_stopvandalizing | 6  | [0]      | 0.005179080887  | 0.004880861258  | 0.0005268546776 |
| contains_harassme        | 7  | [0]      | 0.0002435460302 | 0.0002435460302 | 0.0002435460302 |
| contains_willreport      | 8  | [0]      | 1.99E-05        | 1.99E-05        | 1.49E-05        |
| contains_email           | 9  | [0]      | 0.00224658787   | 0.001998071513  | 0.001068620337  |
| contains_url             | 10 | [0]      | 0.03810749824   | 0.03216298697   | 0.01740111534   |
| keyword_toxic_stopwords  | 11 | [1]      | 0.2880105769    | 0.2505840134    | 0.2215821545    |
| keyword_please           | 12 | [0]      | 0.3814179349    | 0.2874191079    | 0.1458443095    |
| keyword_thanks           | 13 | [0]      | 0.1177719017    | 0.08879986481   | 0.03878843305   |
| capslock                 | 14 | [1]      | 0.01294273189   | 0.008543992366  | 0.005193991869  |



Here is a graph to get a sense of the overall coverage of these LFs together.

![Image with caption](https://drive.google.com/uc?export=view&id=1-SY4k42fDYtE7wvkZf49W6bRMOOgQnNi "coverage")

Our next step is to convert these numerous labels into a final set of probabilistic noise-aware labels.

# Training a Generative Model Using Label Matrix
This Label Matrix is all that we need to train a generative model to obtain a final single set of probabilistic labels. It estimates the accuracies of the various LFs, accounting for any potential correlations between them, and how often they label vs abstain. Note that no gold labels are used during the training process. The only information we need is the label matrix, which contains the output of the LFs on our training set. Snorkel’s [LabelModel](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.label_model.LabelModel.html#snorkel.labeling.model.label_model.LabelModel) is able to learn weights for the labeling functions using only the label matrix as input. 
Before we train the generative model, let’s briefly discuss the baseline method -  [MajorityLabelVoter](https://snorkel.readthedocs.io/en/master/packages/_autosummary/labeling/snorkel.labeling.model.baselines.MajorityLabelVoter.html#snorkel.labeling.model.baselines.MajorityLabelVoter) to combine the results of the LFs. In this method, if more LFs voted “toxic”, we treat comments to be toxic and vice-versa. Ideally, the LFs should not be treated identically -- we saw in my previous post that they may be correlated and if we consider a majority vote, some signals might be overrepresented. In the LFs defined above, better_profanity and keyword-based profanity lookup are likely to label a large number of common data points. More generally, we need to denoise the LFs and MajorityLabelVoter does not help with that.

The LabelModel is able to denoise the LFs, estimate their accuracies and weights to output a single set of noise-aware confidence-weighted labels. Notice in the graph below that the labels are probabilistic in nature.

![Image with caption](https://drive.google.com/uc?export=view&id=11u7mEHiLOfK3xddAn4hRTApAPKe_YB_c "labelmodeloutout")

There would still be many comments that cannot be labeled with LFs. In this case, 73.25% of the train set is labeled by one or more LFs. We will use these labels to train a discriminative model for toxic comment classification. 

At this point, we can also apply LFs to the validation set and inspect the coverage and empirical accuracies associated with each LF. Based on these we can rethink some of our LFs, and re-apply them to the train set before we train a discriminative model. It’s important to keep in mind a good balance of coverage and accuracy. An LF should label as much of the data as possible, and as accurately as possible.

|                          | j  | Polarity | Coverage        | Overlaps        | Conflicts       | Correct | Incorrect | Emp. Acc.    |
|--------------------------|----|----------|-----------------|-----------------|-----------------|---------|-----------|--------------|
| contains_work_of_art     | 0  | [0]      | 0.08185721954   | 0.07711576311   | 0.04195741635   | 877     | 38        | 0.9584699454 |
| contains_entity          | 1  | [0]      | 0.3520307747    | 0.2842190016    | 0.1549472177    | 3705    | 230       | 0.9415501906 |
| textblob_polarity        | 2  | [0]      | 0.003936303453  | 0.003220611916  | 0.0006262300948 | 41      | 3         | 0.9318181818 |
| textblob_subjectivity    | 3  | [0]      | 0.1148684917    | 0.07058507783   | 0.03399534801   | 1016    | 268       | 0.7912772586 |
| contains_profanity       | 4  | [1]      | 0.08427267848   | 0.08248344963   | 0.05457147969   | 588     | 354       | 0.6242038217 |
| contains_pleaseread      | 5  | [0]      | 0.005457147969  | 0.005457147969  | 0.001162998748  | 59      | 2         | 0.9672131148 |
| contains_stopvandalizing | 6  | [0]      | 0.005546609411  | 0.005188763643  | 0.0005367686527 | 59      | 3         | 0.9516129032 |
| contains_harassme        | 7  | [0]      | 0.0003578457685 | 0.0003578457685 | 0.0003578457685 | 2       | 2         | 0.5          |
| contains_willreport      | 8  | [0]      | 8.95E-05        | 8.95E-05        | 0               | 0       | 1         | 0            |
| contains_email           | 9  | [0]      | 0.003310073358  | 0.003131150474  | 0.001431383074  | 36      | 1         | 0.972972973  |
| contains_url             | 10 | [0]      | 0.0397208803    | 0.03453211666   | 0.01851851852   | 424     | 20        | 0.954954955  |
| keyword_toxic_stopwords  | 11 | [1]      | 0.2912864555    | 0.2536231884    | 0.2252639113    | 730     | 2526      | 0.2242014742 |
| keyword_please           | 12 | [0]      | 0.3956879585    | 0.2964752192    | 0.152979066     | 4158    | 265       | 0.9400859145 |
| keyword_thanks           | 13 | [0]      | 0.115136876     | 0.08400429415   | 0.03685811415   | 1261    | 26        | 0.9797979798 |
| capslock                 | 14 | [1]      | 0.01315083199   | 0.008677759885  | 0.005546609411  | 81      | 66        | 0.5510204082 |


From the table above, we can see that some LFs worked a lot better than others. LF contains_entity has pretty good coverage of 35%, and high accuracy of 94%. The LF keyword_thanks has the highest accuracy (97.9%) on the validation set with 11% coverage. contains_pleaseread has low coverage, but very good accuracy. It seems like we can discard contains_willreport from our pool of LFs in future iterations as it has extremely low coverage. LF keyword_toxic_stopwords, which has high coverage, seems to get more labels incorrect than correct. We might need to further inspect this list of words for future iterations.

The output from the LabelModel we trained has a 74.8% accuracy on the validation set of 11K comments. 

# Training a Discriminative Model Using Probabilistic Labels
Finally, we will then use these probabilistic labels to train a binary classifier for toxic comment classification. This is important so that we can generalize beyond what has been labeled using labeling functions. The discriminative model needs to be able to support probabilistic labels.

I trained a Logistic Regression model using bag of n-grams features and saw an accuracy of 90.5% on the validation set of 11K comments.

<!-- and BERT features - -->

<!-- | features                 | accuracy                       |
|--------------------------|--------------------------------|
| bag of n-grams           | 90.5%                          |
| BERT                     | 1                              | -->


<!-- and obtained a validation set of 11K comments with an accuracy of 90.5%.  -->

# Future Work
A lot more can be done with labeling functions. Here are a couple of other ideas and resources if readers would like to extend this work - 
* Look for well known [homonyms](https://en.wikipedia.org/wiki/Homonym) - words that share the same spelling, regardless of pronunciation, particularly those that represent slurs in a different culture
* Expand the list of known profane words with the help of numerous available resources such as [this](https://data.world/wordlists), [this](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt) or [this](https://data.world/natereed/banned-words-list). The latter two are English-only.
* [Hatebase’s blog](https://hatebase.org/news/2019/01/31/pilotfish-with-eggplant) has numerous useful insights on what are the tricky cases to look for toxicity through emoji which can be perceived as offensive, as well as xenophobic references, including allusions to social parasitism, attempts at dehumanization, allusions to white supremacy, etc. I already wrote some keyword search LFs with the keywords on the Hatebase’s blogs, but with only a few keywords, coverage of the LFs is pretty limited.

I would personally learn more about this domain and write more number of rich labeling functions as it seems like this could be a good sustainable approach to monitoring toxicity on the internet.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This post can be cited as:

```
@article{neeraj2020wsfortoxicity,
    title = "Data Labeling using Weak Supervision: In Action",
    author = "Neeraj, Trishala",
    journal = "trishalaneeraj.github.io",
    year = "2020",
    url = "https://trishalaneeraj.github.io/2020-07-26/data-labeling-weak-supervision"
}
```