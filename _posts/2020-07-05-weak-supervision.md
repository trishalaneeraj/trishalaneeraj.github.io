---
layout: post
title: "Weak Supervision for Online Discussions"
author: "Trishala Neeraj"
---

When working on the data I used for my previous blog post, I grew particularly interested in learning how the dataset was labeled for toxicity and identities. I believe that understanding how this data was annotated and curated is essential and will help in thinking more clearly about the end goal of effectively and accurately identifying toxic comments allowing for safer, constructive, and more inclusive online discussions.

Jigsaw released a labeled [dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) that comprises of labeled comments published by Civil Comments along with labels from almost 9000 individual annotators -- approximately 2 million comments labeled for toxicity. Additionally, approximately 250,000 comments are labeled by human annotators for references of identities (i.e. sexual orientation, religion, race/ethnicity, disability, and mental illness). Each comment is shown to multiple human annotators so as to better capture the subjectivity of the problem by allowing multiple voices to be weighed into the label. Human annotators can disagree on the label of different comments based on their perspectives, experiences, and background. The team at Jigsaw discussed the idea of weighing individual raters differently based on who is best suited to rate a comment. More details in their blog post [here](https://medium.com/the-false-positive/creating-labeled-datasets-and-exploring-the-role-of-human-raters-56367b6db298).

Here’s an example comment from the dataset 
> 'I believe  that any serious student of history or current events should come to the same conclusion of George Washington that party politics are an abomination.    From his Farewell Address:  “However [political parties] may now and then answer popular ends, they are likely in the course of time and things, to become potent engines, by which cunning, ambitious, and unprincipled men will be enabled to subvert the power of the people and to usurp for themselves the reins of government, destroying afterwards the very engines which have lifted them to unjust dominion.” \n\nHis only blind spot is that he failed to predict that by 2016 there would also be "cunning, ambitious, and unprincipled "  women involved as well.'”

This comment was shown to 5 annotators who label it for different aspects and degrees of toxicity. The “worker” column represents unique worker (annotator) IDs. From the data, it looks like 1 out of 5 annotators thought this comment was toxic for online discussions.

![Image with caption](https://drive.google.com/uc?export=view&id=15IxwwwCoiA-7uhnY3a05vo0F85x1khRY "Toxicity labeled")

The same comment was also shown to 10 annotators for labels for identity - 

![Image with caption](https://drive.google.com/uc?export=view&id=1TiZdfX4jLwi5KSRKhX_wDrwxqL3GGhrF "Identity labeled")

The responses can be aggregated appropriately to produce a final label to train a machine learning model to predict the probability that a human will find the comment toxic. 

I had a few thoughts and questions about this process for data collection.

* How can we create a high-quality labeled dataset at scale? Would this approach of utilizing crowdwork be sustainable long-term? Do all comments need careful review by multiple human annotators? Or are some harder to label than others and therefore must receive special attention? Is there a way to determine these data points intelligently?

* How is this annotation work affecting those reading and analyzing them? It’s possible that annotators are having to repeatedly read and analyze comments targeted towards the underrepresented group they identify with. This is a real problem and has been [discussed](https://twitter.com/kirbyconrod/status/1139184438846402562) by researchers in the field. How can we make this process better for annotators?

* How high-quality are these labels produced by crowdwork? When working on this [paper](https://maxiao.info/papers/ma2017computational.pdf), I learned that more often than not we’d need to further employ measures for quality control in annotated data. We received responses from Amazon Mechanical Turk workers and had to filter out potential “spammers” using metrics such as task completion time, the standard deviation and mean of the ratings of the same annotator, and also answers to a linguistic attentiveness question.

A couple of weeks ago, I picked up Robert Munro’s book [Human‑in‑the‑Loop Machine Learning](https://livebook.manning.com/book/human-in-the-loop-machine-learning/welcome/v-7/) to learn more about annotation as a science. The author describes how quickly it can become intractable to annotate each data point with the help of humans, and how active learning fits into the picture. I learned about “Repetition Priming” - when the sequence of data samples shown to the annotator can impact the results of the annotation process by influencing someone’s perception. In this case, if an annotator happens to be seeing numerous toxic comments one after the other, they’re likely to mistakenly annotate a non-toxic comment as toxic.

Human annotations are expensive, but also not always straightforward to reliably make use of in training machine learning applications. With these considerations, I seek to explore Weak Supervision for the task of labeling this data at scale.

# Weak Supervision

Weak Supervision focuses on capturing supervision signals from subject matter experts at a higher abstraction level, for example, in the form of heuristics, expected data distributions, or constraints. In [Data Programming: Creating Large Training Sets, Quickly](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly.pdf) (Ratner et al, 2016), authors propose Data Programming, a simple, unifying framework for weak supervision, for the programmatic creation and modeling of training datasets. The labels generated through this are noisy and from multiple overlapping sources. We generate labels through this by defining numerous “labeling functions” which capture domain expertise -- in the form of heuristics, existing knowledge bases (distant supervision), etc. Each labeling function would label a subset of the data -- i.e., label or abstain from labeling depending on the case. Many labeling functions together would be able to label a large proportion of the data, although they may overlap in coverage. That is, a single data point may be labeled by multiple labeling functions.

For the task at hand, what are some signals to construct labeling functions for toxicity? According to the Jigsaw Team, toxicity is defined as anything rude, disrespectful, or otherwise likely to make someone leave a discussion.

I studied a sample of data to develop initial ideas on the kind of heuristics that would help in writing these labeling functions. One reasonable signal to look for in this use case could be the use of profanity in a comment. It is likely to make someone uncomfortable and leave the online discussion, and hence the comment could be labeled as toxic. This may not always be strictly true depending on the context of the conversation, and hence a weak heuristic to start with. Using an external knowledge base such as one found [here](https://code.google.com/archive/p/badwordslist/downloads) could be a good starting point. Of course, creating a custom list for this specific use case would lead to more favorable results, and is ideally built with the help of someone studying online abuse and toxicity in depth.

Another heuristic I can think of is that non-toxic comments are likely to contain words like thank you and please. Again, usage could be sarcastic or simply not be well-intentioned. Still, something to get the ball rolling. A third one could be that comment written in all-caps might indicate strong emotion, likely negative and toxic.

The 3 heuristics described above can be used to write labeling functions which will have widely varying error rates and may conflict on certain data points. This is similar to the inter-annotator agreement when employing human annotators for tasks like this.

The authors of this paper (Ratner et al 2016) show that by explicitly representing this as a generative process, we can “denoise” the labels generated by learning the accuracies of the labeling functions and how they’re correlated. Some commonly occurring types of dependencies include: similar, fixing, reinforcing, exclusive. Consider the following situation given 2 labeling functions LF1 and LF2:
* LF2 labels only when LF1 labels
* LF1 and LF2 disagree on their labeling 
* LF2 is actually correct

Here, LF2 fixes the mistakes that LF1 makes, and hence these two LFs can have a “fixing” dependency.

Once we model, denoise, and unify these noisy, conflicting, and potentially correlated labeling functions, we obtain a final probabilistic label set for the data, which can finally be used to train a noise-aware Discriminative Model.

![Image with caption](https://drive.google.com/uc?export=view&id=1o-aZRds5-6ctAAXzW57td9H9UdH1sJu2 "Weak Supervision Overview")

The authors use this generative model to optimize the loss function of the discriminative model we want to train. One of the key theoretical results of this paper is that we can achieve the same asymptotic scaling as supervised learning methods, but that scaling depends on the amount of unlabeled data and uses only a fixed number of labeling functions.

In future work, I'll share more details on the labeling functions I wrote for this task, along with details on how much data they're able to label and how much they agree / conflict with each other.