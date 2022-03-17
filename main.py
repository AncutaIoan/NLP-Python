import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec #Word2Vec is mostly used for huge datasets

train = pd.read_json(r"News_Category_Dataset_v2.json", lines = True)

#for i in range(10000):
 #   print(train["category"][i], "-", train["headline"][i])

def dummy_preprocess(text):
    text2=[]
    for i in range(10000):
        text2.append(text[i].lower())
    return text2
def dummy_tokenize(text):
    text2 = []
    for i in range(10000):
        text2.append(text[i].split(" "))
    return text2


test_preprocess = dummy_preprocess(train["headline"])
test_tokens = dummy_tokenize(test_preprocess)
#print(test_tokens)

import spacy
nlp = spacy.load('en_core_web_sm')
stop_words_spacy = nlp.Defaults.stop_words
all_words_without_stops=[]
for i in test_tokens:
    all_words_without_stops.extend([word for word in i if word not in stop_words_spacy])

import string

string.punctuation
test_nopct=[]
for i in test_tokens:
    test_nopct.extend([test.translate(str.maketrans('', '', string.punctuation)) for test in i])

xtrain = test_tokens[:8000]
xtesting = test_tokens[8000:]
ytrain = []
for i in range(8000):
    ytrain.append([train["category"][i]])
ytest = []
for i in range(8000,10000):
    ytest.append([train["category"][i]])
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

wl = WordNetLemmatizer()
model = Word2Vec(test_tokens,min_count=1)  #min_count=1 means word should be present at least across all documents,
#if min_count=2 means if the word is present less than 2 times across all the documents then we shouldn't consider it


w2v = dict(zip(model.wv.index_to_key, model.wv))  #combination of word and its vector

#for converting sentence to vectors/numbers from word vectors result by Word2Vec
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


X_train, X_val, y_train, y_val = train_test_split(test_tokens,
                                                  train["category"][:10000],
                                                  test_size=0.2,
                                                  shuffle=True)
X_train_tok = []
X_val_tok = []
for i in X_train:
    X_train_tok.append([nltk.word_tokenize(j) for j in i])
for i in X_val:
    X_val_tok.append([nltk.word_tokenize(j) for j in i])

X_train_vectors_tfidf=[]
X_val_vectors_tfidf=[]
for i in X_train:
    X_train_vectors_tfidf+=(tfidf_vectorizer.fit_transform(j for j in i))
for i in X_val:
    X_val_vectors_tfidf+=(tfidf_vectorizer.transform(
        j for j in i))   # Don't fit() your TfidfVectorizer to your test data: it will

modelw = MeanEmbeddingVectorizer(w2v)
X_train_vectors_w2v = modelw.transform(j for j in test_tokens[:8000])
X_val_vectors_w2v = modelw.transform(j for j in test_tokens[8000:10000])

lr_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')

lr_tfidf.fit(X_train_vectors_tfidf, ytrain)  # de aici nu a mai mers ( uneori merge, alteori nu)

# Predict y value for test dataset
y_predict = lr_tfidf.predict(X_val_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_val_vectors_tfidf)[:, 1]

print(classification_report(y_val, y_predict))
print('Confusion Matrix:', confusion_matrix(y_val, y_predict))

fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)