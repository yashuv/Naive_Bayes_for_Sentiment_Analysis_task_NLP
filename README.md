# Naive Bayes for Sentiment Analysis task NLP
We use another useful method for classification called Naive Bayes. It is an example of supervised machine learning and shares many similarities with the logistic regression method I used previously.

We will learn the theory behind Naive Bayes' rule for conditional probabilities, then apply it toward building 
a Naive Bayes tweet classifier of our own! Given a tweet, we will decide if it has a positive sentiment or a negative one. Specifically we will: 

* Train a naive bayes model on a sentiment analysis task
* Test using your model
* Compute ratios of positive words to negative words
* Do some error analysis
* Predict on our own tweet

I achieved the following learning goals:
* Error analysis
* Naive Bayes inference
* Log likelihood
* Laplacian smoothing
* Conditional probabilities
* Bayes rule
* Sentiment analysis
* Vocabulary creation
* Supervised learning

## Naive Bayes classifier
A **Naive Bayes classifier** is a machine learning algorithm that is used for classification tasks. It is a simple algorithm that is based on the Bayesian theorem. The model is trained on a dataset and then used to predict the class of new data points. The model is called "naive" because it is based on the assumption that all features being used for classification are independent of each other. This assumption is called the independence assumption. This is rarely the case in reality.<br>
There are a few reasons why the Naive Bayes classifier is used for tweets. First, the classifier is very fast and efficient, which is important when dealing with large amounts of data. Second, the classifier is relatively simple and easy to understand, which is important when trying to explain the results to others. Finally, the Naive Bayes classifier typically performs well on tweet data.

## Conditional probability
The **Conditional probability** is the probability that an event will occur given that another event has already occurred.
<pre>      P(A | B) = P(A ∩ B) / P(B)
	  P(B | A) = P(B ∩ A) / P(A)
</pre>
where; P(A | B) is the conditional probability of an event A, given that event B has occurred, is the probability that A will occur given that B has already occurred.<br>
Conditional probability is used in the calculation of the naive bayes algorithm. If we are using conditional probability to calculate the probabilities for our naive bayes model, then higher conditional probabilities will generally result in a better model.

## Bayes rule
The **Bayes rule** is based on the mathematical formulation of conditional probabilities.
<pre>
       P(A | B) = P(B | A) * P(A) / P(B) 
</pre>
That's with Bayes rule, you can calculate the probability of A given B if you already know the probability of B given A and the ratio of the probabilities of A and B.

We usually compute the probability of a word given a class as follows:
<pre>
    P(w<sub>i</sub>|class) = freq(w<sub>i</sub>, class) / (N<sub>class</sub> ; <i>class</i> ∈ {Positive, Negative}
</pre>
However, if a word does not appear in the training, then it automatically gets a probability of 0, to fix this we add smoothing.

## Laplace smoothing
**Laplace smoothing** is used in naive bayes classifier to avoid zero probabilities. This is done by adding a small number to each probability, which prevents zero probabilities from occurring.
<pre>
    P(w<sub>i</sub>|class) = (freq(w<sub>i</sub>, class) + 1) / (N<sub>class</sub> + V)
</pre>
Note that we added a 1 in the numerator, and since there are V words to normalize, we add V in the denominator.<br>
<i> freq(w<sub>i</sub>, class)</i>: frequency of word w in class.<br>
N<i><sub>class</sub></i>: frequency of all words in class.<br>
<i>V</i>: number of unique words in vocabulary.<br>
<i>class</i> ∈ {Positive, Negative}

## Log likelihood
The **log likelihood** is a measure of how likely a given data point is to belong to a given class, given a set of features. It is used in Naive Bayes tweet classification to calculate the score that will allow us to decide whether a tweet is positive or negative. It is a measure of the goodness of fit of the model to the data. The higher the ratio, the more positive the word is.

<pre>log likelihood = λ(w<sub>i</sub>) = log (P(w<sub>i</sub>|pos) / P(w<sub>i</sub>|neg)) </pre>
Having the *λ* dictionary will help a lot when doing inference. <br>
If *Σ log likelihood<sub>i</sub> > 0* (positive), we will classify the document to be positive. If we got a negative number we would have classified it to the negative class. 

## Training Naïve Bayes
To train your naïve Bayes classifier, you have to perform the following steps:<br>
1. **Get or annotate a dataset with positive and negative tweets**
2. **Preprocess the tweets**
	* Lowercase
	* Remove punctuation, urls, names
	* Remove stop words
	* Stemming
	* Tokenize sentences
3. **Compute freq(w, class)**
4. **Get** P(w|pos),P(w|neg)
5. **Get** λ(w)
	λ(w) = log(P(w|pos)/P(w|neg))
6. **Compute** logprior = log(P(pos)/P(neg))
	logprior = log(D<sub>pos</sub>/D<sub>neg</sub>), where D<sub>pos</sub> and D<sub>neg</sub> correspond to the number of positive and negative documents respectively. 
	
## Testing Naïve Bayes
Once the model is trained, we will use it to test the model on real test examples by predicting the sentiments of new unseen tweets and evaluating the model's performance by looking at the accuracy, precision, and recall of the model.<br>

The steps generally followed are as follows:
1. Prepare a set of tweets, including at least two from each of the classes you wish to test (e.g., positive, negative, neutral).
2. Preprocess the tweets, including stemming, lemmatization, and removing stopwords and other noise.
3. Run the tweets through the classifier and predict the results.
4. Compare the results to the actual classes of the tweets to the evaluate the performance of model.<br>
	` Error is the average of the absolute values of the differences between predicted result(y_hats) and actual class(test_y).`
	<pre> Accuracy = 1 - Error </pre>
	
## Applications of Naïve Bayes
- Text classification
- Author Identification
- Spam filtering
- Sentiment analysis
- Information retrieval
- Word disambiguation
- Recommendation systems
- Predicting the weather

This method is easy to implement and computationally efficient, especially for high-dimensional data. It is highly scalable. It can be used for both binary and multi-class classification. It is often successful in classification tasks even when the underlying assumptions are not perfectly met.

## Conclusion
I learned the theory behind Naive Bayes' rule for conditional probabilities, then apply it toward building a Naive Bayes tweet classifier of my own.<br>
I had the chance to implement all of the concepts and skills that we spoke about. I also learned how to do error analysis, which is an important tool that allows us to debug your model if it's not working.

<br>
Thank you, and happy learning!<br>

---
