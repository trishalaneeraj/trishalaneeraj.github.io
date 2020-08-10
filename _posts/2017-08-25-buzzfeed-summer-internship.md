---
layout: post
title: "Summer 2017 @BuzzFeed Data Science"
author: "Trishala Neeraj"
---

> This blog post was originally published by me on Medium and can be found [here](https://tech.buzzfeed.com/how-we-tagged-14-000-buzzfeed-quizzes-using-k-means-clustering-95fc46bc6daf). It describes my project at BuzzFeed during the period of my summer internship.

At BuzzFeed, we have lately been concerned with the quality of the metadata related to our Quizzes. In the interest of our users being able to discover our best content, we decided to redo the tagging system.

Currently, we have an author-provided set of tags for each of the Quizzes and while they are good tags, we felt the need to automate this process for the sake of consistency. So, in the past few weeks we worked on automating the process of generating low-level consistent tags. That means we won’t just have tags like ‘Food’ or ‘Trivia’, we will also use ‘Pizza’, ‘Geography Trivia’, ‘Who Said It’, etc.

After some exploratory analysis of the available data, we landed on Clustering. This blog post will be a walk-through of the technical steps involved in this task. We will start by giving a refresher on clustering and K-Means Clustering algorithm, which also includes some details of implementation, and then explain how exactly we prepared our data for clustering. Towards the end we’ll briefly touch upon how to interpret the resulting clusters.

**Why are we clustering?**

Like we mentioned earlier, we are looking for a whole new set of rather low-level tags. We don’t just want to know that a quiz was about movies, we also want to know if it was about animated movies. Since we don’t have our quizzes tagged this way already and there’s no ‘ground truth’ or labels we want our algorithm to learn from, we’re going to be looking at [unsupervised machine learning](https://en.wikipedia.org/wiki/Unsupervised_learning) algorithms. [Clustering](https://en.wikipedia.org/wiki/Cluster_analysis) is the most popular approach among those. It is the task of assigning a set of data points or observations into subsets called clusters such that the observations within the same cluster are similar with respect to a predefined metric and the observations in different clusters are dissimilar. While there are several ways to go about clustering, we choose K-Means Clustering because it scales well to large number of samples.

**What is K-Means Clustering?**

K-Means is an iterative clustering algorithm that partitions a dataset to form coherent subsets of all data. The algorithm iterates between 2 steps — the _cluster assignment_ step and the _move centroid_ step. Here is an animation [link](http://shabal.in/visuals/kmeans/5.html) that will give a better insight about the iteration process. The optimization objective of the algorithm is:

<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/312/1\*JWBgHUnf-CRusO3yU74IIA.gif" width="156" height="51"/> -->

![Image with caption](https://drive.google.com/uc?export=view&id=1gLAtQAQvb8bEWV2pf0M3B9Og2fKJ38gS "Overview")

This is the average squared distance between each data point and the location of the cluster centroid to which it has been assigned. c is the index to which the data point is currently assigned. This algorithm, through several iterations, tries to find c and to minimize the above cost function. For overall conceptual clarity, this [video](https://www.coursera.org/learn/machine-learning/lecture/93VPG/k-means-algorithm) might be helpful.

We used the [scikit-learn implementation](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) of K-Means which is widely used and well-documented.


{% highlight python %}from sklearn.cluster import KMeans  
model = KMeans(n_clusters=40, init='k-means++', max_iter=300, n_init=50, random_state=3)  
model.fit(data)
{% endhighlight %}

It is worth discussing the parameters briefly:

1.  n_clusters — the number of clusters we want, which needs to be declared upfront
2.  init — the method for initialization. ‘K-means++’ selects initial cluster centers in such a way that they are distant from each other. This allows quick convergence and better results compared to completely random initialization.
3.  max_iter — the maximum number of iterations (assign and move steps) each initialization should go through.
4.  n_initialization — the number of different initializations given, that is the number of times the algorithm will be run with different centroid seeds. Depending on initial conditions, different clusters are formed. The final results will be the best output of _n_init_ consecutive runs in terms of inertia.
5.  random_state — a random seed for reproducibility.

The remaining part of this post will focus on how to prepare the ‘data’ that the model is fit on, and how to obtain and interpret the predictions.

**What is the training data?**

We fed the algorithm on the textual content of BuzzFeed Quizzes. We decided to extract features specifically from the following title, blurb and author-provided tags. For example, for the following quiz:

<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/2524/0\*NTNQIr3mGMfkSpkZ." width="1262" height="310" srcSet="https://miro.medium.com/max/552/0\*NTNQIr3mGMfkSpkZ. 276w, https://miro.medium.com/max/1104/0\*NTNQIr3mGMfkSpkZ. 552w, https://miro.medium.com/max/1280/0\*NTNQIr3mGMfkSpkZ. 640w, https://miro.medium.com/max/1400/0\*NTNQIr3mGMfkSpkZ. 700w" sizes="700px"/>
 -->
![Image with caption](https://drive.google.com/uc?export=view&id=1ZNm-AA9QKNCZCpjXEtN4B4hnG1SVO6u8 "Overview")

Title

<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/2560/0\*CtgePQsb8QcZBCNS." width="1280" height="280" srcSet="https://miro.medium.com/max/552/0\*CtgePQsb8QcZBCNS. 276w, https://miro.medium.com/max/1104/0\*CtgePQsb8QcZBCNS. 552w, https://miro.medium.com/max/1280/0\*CtgePQsb8QcZBCNS. 640w, https://miro.medium.com/max/1400/0\*CtgePQsb8QcZBCNS. 700w" sizes="700px"/>
 -->
![Image with caption](https://drive.google.com/uc?export=view&id=1B63YyZl_S4JQVtMSuesKkdqOXzWUc_Pw "Overview")

Blurb
<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/1348/0\*r6YgkVFmKJAu9wii." width="674" height="60" srcSet="https://miro.medium.com/max/552/0\*r6YgkVFmKJAu9wii. 276w, https://miro.medium.com/max/1104/0\*r6YgkVFmKJAu9wii. 552w, https://miro.medium.com/max/1280/0\*r6YgkVFmKJAu9wii. 640w, https://miro.medium.com/max/1348/0\*r6YgkVFmKJAu9wii. 674w" sizes="674px"/> -->
![Image with caption](https://drive.google.com/uc?export=view&id=1dPvD3vS2mF0F-J-z2I_WfiE2UpHEMUDt "Overview")

And lastly, author provided tags in the existing metadata

<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/914/0\*SPCywe1wXQmauB4V." width="457" height="27" srcSet="https://miro.medium.com/max/552/0\*SPCywe1wXQmauB4V. 276w, https://miro.medium.com/max/914/0\*SPCywe1wXQmauB4V. 457w" sizes="457px"/> -->
![Image with caption](https://drive.google.com/uc?export=view&id=1RNk2tPpxnSpTt2untzlK3wpHckQRa_HV "Overview")


The above is not a content-related tag, but simply an indication of where it was posted on the website. Other examples of author provided tags are ‘celebrity’, ‘movies’, etc.

At a later stage we will also incorporate questions, but as of now features from questions did not appear very informative. This is mainly because the format in which questions are present in our metadata is highly varied. In the older quizzes we only had questions in the form of images instead of text, and some times text was embedded within images. Also, a huge chunk of questions did not convey much out of context. For example, ‘Pick a shoe’ or ‘Does this make you laugh?’ couldn’t really be associated with a topic or a type of quiz. They could be a part of any quiz.

Now, we have all the text. We’re not ready yet! As expected, there are all sorts of abbreviations, punctuation and special characters in our quizzes, and we want to clean it up. Additionally, we’re going to perform some standard text preprocessing like removing special characters, lower-casing, removing stop words and lemmatizing. This will give us clean sequences of words.

**Feature Extraction**

We cannot input text sequences we generated above into our algorithm. Algorithms, in general, expect numerical or vector features, especially of fixed shape. So, we will represent our words as vectors, or in other words, ‘vectorize’ them. The approach for ‘vectorization’ includes tokenizing (giving an integer ID to each ‘token’ separated by space or punctuations), counting (frequency of occurrences of these tokens), and lastly normalizing (giving weights to tokens based on the frequency of occurrences). For our text data, we used [term frequency-inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (tf-idf) features which allows us to weigh terms appropriately. Words that occur too often and don’t mean much, are given less weight and vice versa.

On analyzing our intermediate results, we noticed that ‘new’ from ‘New York’ and ‘Orange Is The New Black’ was not seen any differently by our algorithm due to tokenization. Thus, we decided to vectorize phrases, i.e., groups of words that co-occur often, rather than vectorizing each word independently. This was particularly useful in looking at [named-entities](https://en.wikipedia.org/wiki/Named_entity) as one token. For example, ‘New_York’, ‘Big_Brother’, ‘Harry_Potter’ etc.

We have used a combination of [NLTK](http://www.nltk.org/), [gensim](https://radimrehurek.com/gensim/models/phrases.html#module-gensim.models.phrases) and [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) (all open source packages) to tokenize, detect phrases and vectorize. Below is the code snippet:

{% highlight python %}def feature_extraction(preprocessed_text):  
    sentence_stream = [nltk.word_tokenize(each) for each in preprocessed_text]  
    phrases = Phrases(sentence_stream)  
    new_sentence_stream = [phrases[each] for each in sentence_stream] #tokenized words/phrases  
    input_text = [" ".join(each) for each in new_sentence_stream] #preparing for vectorization  
    vectorizer = TfidfVectorizer(analyzer = "word", \\  
                             tokenizer = None, \\  
                             preprocessor = None, \\  
                             stop_words = None,\\  
                             max_features = 6000)  
    tfidf_matrix = vectorizer.fit_transform(input_text) # sparse  
    tfidf_matrix = tfidf_matrix.toarray() #dense  
    return vectorizer, tfidf_matrix  
vectorizer, data = feature_extraction(clean_text)
{% endhighlight %}

The resulting tfidf_matrix is the data we fit our model on. The _max_features_ parameter is used to set the maximum vocabulary size, i.e., it sets the dimensions of our feature space — we’re now looking at a 6000 dimensional space. So, tfidf_matrix we obtained has the dimensions number-of-quizzes x _max_features_.

**Understanding the results**

Next, we move onto understanding the results from fitting the model and making predictions.

One of the attributes of the fit model is to be able to see the exact coordinates of cluster centers (see Figure 1).
<!-- 
<img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/2540/0\*SXzTD5nddlYh2OEa." width="1270" height="554" srcSet="https://miro.medium.com/max/552/0\*SXzTD5nddlYh2OEa. 276w, https://miro.medium.com/max/1104/0\*SXzTD5nddlYh2OEa. 552w, https://miro.medium.com/max/1280/0\*SXzTD5nddlYh2OEa. 640w, https://miro.medium.com/max/1400/0\*SXzTD5nddlYh2OEa. 700w" sizes="700px"/>
 -->
![Image with caption](https://drive.google.com/uc?export=view&id=1KMOuYP3YC-UfksLwBPT6hRJG84Ep-RAe "Overview")

Remember the max_features parameter we briefly discussed above? We set it to 6,000. That means that we’re looking at a 6,000 dimensional space and the clusters we obtained are found in this feature space. The dimensions of the above array is 40x6,000.

We can use [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) to find the words and phrases that lie closest to the cluster centers. Then we can determine the main topic of the cluster and give it a name. To do that, our first step would be to sort the index cluster centers in descending order.

{% highlight python %}
ordered_centroids = model.cluster_centers_.argsort()[:, ::-1]
{% endhighlight %}

Then we obtain the top terms in each cluster by finding out which terms are closest to the centroids. Below is a snippet for obtaining top 60 terms in each of the cluster:

{% highlight python %}for idx in ordered_centroids[i, :60]:  
    words.append(vectorizer.get_feature_names()[idx])
{% endhighlight %}

If we print those terms, it would look something like this:

<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/2560/0\*ue0ZtTtEk4lcdBPr." width="1280" height="164" srcSet="https://miro.medium.com/max/552/0\*ue0ZtTtEk4lcdBPr. 276w, https://miro.medium.com/max/1104/0\*ue0ZtTtEk4lcdBPr. 552w, https://miro.medium.com/max/1280/0\*ue0ZtTtEk4lcdBPr. 640w, https://miro.medium.com/max/1400/0\*ue0ZtTtEk4lcdBPr. 700w" sizes="700px"/> -->
![Image with caption](https://drive.google.com/uc?export=view&id=1XUaphKlqCnd7jR-RvPhT6RYSWhggSu43 "Overview")

In this cluster, the term ‘music_video’ is the closest to the cluster center of Cluster 0. ‘identify’, ‘youtube_comment’, etc. are the next few closest words. Recall that some terms are actually 2 words connected by an underscore because they co-occur very frequently.

Printing the top terms is very important since that will allow us to assign a label to all of these clusters. Cluster 0 above appears to be mostly about music videos. Similarly, if we print out terms in all of the clusters we created, we can attach each cluster with a label.

**How many clusters?**

We kept this one for the last because the answer is rather unsatisfying. In all fairness, this is one of the most challenging parts about K-Means. K-Means is a computationally difficult problem ([NP-Hard](https://en.wikipedia.org/wiki/NP-hardness)). It does converge to a local optimum fairly quickly, though there is no guarantee that the global optimum is found.

We did experiment with a few methods of fixating on the number of clusters, like [silhouette coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering)) and [elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)), but found manual inspection to be the best. We have currently identified 40 clusters.

Again, since this is an unsupervised learning method, we cannot measure performance using metrics like accuracy (we don’t have ground truth values). We asked the quiz writers to validate a small sample of quizzes from each cluster with a google form.

**Visualizing clusters**

The 40 clusters we created existed in a 6000 dimensional space. We needed to perform dimensionality reduction so that we could visualize the clusters. We used [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) for projecting them to 2 dimensional space. Through a zoom of the visualization we created, we would like to draw your attention to 3 clusters — ‘TV Shows’, ‘Nostalgia’ and ‘Disney’. It is easy to see how similar clusters are closer to each other.

<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/1200/0\*yx_nBgFW0sH2-x74." width="600" height="378" srcSet="https://miro.medium.com/max/552/0\*yx_nBgFW0sH2-x74. 276w, https://miro.medium.com/max/1104/0\*yx_nBgFW0sH2-x74. 552w, https://miro.medium.com/max/1200/0\*yx_nBgFW0sH2-x74. 600w" sizes="600px"/>

<img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/398/0\*Uy9tYMdqTm365VEA." width="199" height="83"/> -->
![Image with caption](https://drive.google.com/uc?export=view&id=1BK_vlcueQQIDm5TiVYyM7EvHbHOANEPX "Overview")

**How are we exactly using these clusters?**

Apart from backfilling tags for all our quizzes, and periodically re-clustering to find new tags, we are planning on using these to update the filter tabs on the BuzzFeed Quizzes page.

<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/2560/0\*S1k8BfWeUi3CWk0I." width="1280" height="404" srcSet="https://miro.medium.com/max/552/0\*S1k8BfWeUi3CWk0I. 276w, https://miro.medium.com/max/1104/0\*S1k8BfWeUi3CWk0I. 552w, https://miro.medium.com/max/1280/0\*S1k8BfWeUi3CWk0I. 640w, https://miro.medium.com/max/1400/0\*S1k8BfWeUi3CWk0I. 700w" sizes="700px"/> -->
![Image with caption](https://drive.google.com/uc?export=view&id=1GaXBhxYKH4kwtO_HxVX3MjLvuGqmmbgj "Overview")

We recently included filter tabs on the page so that users can binge on their favorite types of quizzes without spending time going through the whole lot. So if you only like ‘Trivia’ Quizzes, just hit the tab and there you have it. Since that worked out really well, for our users and for us, we plan on taking it one step further, this time with automatically generated tags rather than human annotations and filtering.

Our goal was to be able to update these filters periodically based on what’s trending and what’s evergreen. We particularly wanted them be very specific in nature and not too broad. While ‘Food’ is an acceptable filter, we wanted to be able to give you ‘Pizza’ as a filter. We also wanted to prioritize on filters we know a huge number of our users would be interested in. It would be useful to have a filter dedicated to ‘Game Of Thrones’, for example, but not for ‘Pretty Little Liars’. We could simply put the latter in the ‘TVShows’ filter since it’s not very popular these days, not evergreen like ‘Harry Potter’.

With the new set of tags which are more fine (low-level), we can provide you with even more interesting filters like ‘Weddings’ or ‘Style And Fashion’. We can then potentially update them periodically as we discover new tags via clustering!

<!-- <img alt="Image for post" class="s t u hm ai" src="https://miro.medium.com/max/1000/1\*51PRlhH9zMAwe2V4tJ8Obg.gif" width="500" height="270" srcSet="https://miro.medium.com/max/552/1\*51PRlhH9zMAwe2V4tJ8Obg.gif 276w, https://miro.medium.com/max/1000/1\*51PRlhH9zMAwe2V4tJ8Obg.gif 500w" sizes="500px"/> -->

We surely were entertained creating and analyzing these clusters and hope you all have fun using them to discover our content!

* * *

Check out BuzzFeed's [Tech Blog](https://tech.buzzfeed.com/) and [Twitter](http://twitter.com/buzzfeedexp) to learn more about other projects at BuzzFeed.
