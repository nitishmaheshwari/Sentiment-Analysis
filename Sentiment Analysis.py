'''
ALGORITHM:
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

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier
import re, string, random
stop_words = stopwords.words('english')





def noise_removal(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token in tweet_tokens:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        token = re.sub("(#[A-Za-z0-9_]+)","", token)
        # re (library) is used to search a given string in another string.
        # .sub() method replaces it with an empty string.
        
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token)
            
    return cleaned_tokens



def normalize(tweet_tokens):
    normalized_tokens = []
    for token, tag in pos_tag(tweet_tokens):
    # pos_tag() function returns tweets in (token,tag) format
    # Output of pos_tag() function
    # [ ('#FollowFriday', 'JJ'), ('@France_Inte', 'NNP'), ('@PKuchly57', 'NNP'),
    # ('@Milipol_Paris', 'NNP'), ('for', 'IN'), ('being', 'VBG'), ('top', 'JJ'),
    # ('engaged', 'VBN'), ('members', 'NNS'), ('in', 'IN'), ('my', 'PRP$'),
    # ('community', 'NN'), ('this', 'DT'), ('week', 'NN'), (':)', 'NN') ]
        
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
            
        lemmatizer = WordNetLemmatizer()
        normalized_tokens.append(lemmatizer.lemmatize(token, pos))
        
    return normalized_tokens



def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
# Dictionary is a type of hash-map in python
# Yield is a keyword that is used like return, except the function will return a generator.
        
# Generators are iterators, but you can only iterate over them once. 
# It’s because generators do not store all the values in memory, they generate the values on the fly




# import dataset
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')


# tokenization
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')


# noise removal
positive_cleaned_tokens_list=[]
for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(noise_removal(tokens,stop_words))

negative_cleaned_tokens_list=[]
for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(noise_removal(tokens,stop_words))



# normalization (lemmatization)
positive_normalized_cleaned_tokens_list=[]
for tokens in positive_cleaned_tokens_list:
    positive_normalized_cleaned_tokens_list.append(normalize(tokens))

negative_normalized_cleaned_tokens_list=[]
for tokens in negative_cleaned_tokens_list:
    negative_normalized_cleaned_tokens_list.append(normalize(tokens))



# converting tokens to the dictionary form
positive_tokens_for_model = get_tweets_for_model(positive_normalized_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_normalized_cleaned_tokens_list)

        
        
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
custom_tokens = normalize(noise_removal(word_tokenize(custom_tweet)))
print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))