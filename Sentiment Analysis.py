'''
ALGORITHM:
Install ntlk and download the data

Tokenization -> break sentence into words

Normalization (Lemmatization) -> 
            Stemming and lemmatization are two popular techniques of normalization.
            Stemming is a process of removing affixes from a word.
            Lemmatization normalizes a word with the context of vocabulary and 
            morphological analysis of words in text.
            run, running, runs -> run
            are, is, being -> be

Noise removal -> remove hyperlinks, twitter-mentions, remove punctuations, 
                    convert the text into lowercase etc.

Conversion in dictionary form for further computation -> 

Split Data into train and test sets (70:30) ->

Fitting the model ->

Prediction  ->
'''
# CHECH THE README FILE TO INSTALL NECESSARY PACKAGES AND LIBRARIES

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens



def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


# import dataset
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
#tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]



# tokenization
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')



# normalization and noise removal
stop_words = stopwords.words('english')

positive_cleaned_tokens_list = []
for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

negative_cleaned_tokens_list = []
for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


# converting tokens to the dictionary form
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


# splitting data into train and test set (70:30)
positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]
dataset = positive_dataset + negative_dataset
random.shuffle(dataset)
train_data = dataset[:7000]
test_data = dataset[7000:]


# fitting classifier - Naive Bayes
classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(20))


# Prediciton on custom tweet
custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
custom_tokens = remove_noise(word_tokenize(custom_tweet))
print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))