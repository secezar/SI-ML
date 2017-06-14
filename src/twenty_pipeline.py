'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

# from __future__ import print_function

import os
import sys
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.logistic import LogisticRegression

BASE_DIR = ''
TEXT_DATA_DIR = 'C:\\Users\\Jola\\Desktop\\PROJEKTY\\SI\\SI-ML\\src\\glove.6B\\glove.6B.100d.txt'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 10000
VALIDATION_SPLIT = 0.2

selected_categories = [
    'comp.graphics',
    'rec.motorcycles',
    'rec.sport.baseball',
    'misc.forsale',
    'sci.electronics',
    'sci.med',
    'talk.politics.guns',
    'talk.religion.misc']

newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=selected_categories,
                                      remove=('headers', 'footers', 'quotes'))

newsgroups_test = fetch_20newsgroups(subset='test',
                                     categories=selected_categories,
                                     remove=('headers', 'footers', 'quotes'))

train_texts = newsgroups_train['data']
train_labels = newsgroups_train['target']
test_texts = newsgroups_test['data']
test_labels = newsgroups_test['target']


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer(max_features=10000)),
                      ('clf', MultinomialNB())])
text_clf = text_clf.fit(train_texts, train_labels)
predicted = text_clf.predict(test_texts)
print np.mean(predicted == test_labels)

text_clf = Pipeline([('vect', CountVectorizer(max_features=10000)),
                      ('clf', MultinomialNB())])
text_clf = text_clf.fit(train_texts, train_labels)
predicted = text_clf.predict(test_texts)
print np.mean(predicted == test_labels)

text_clf = Pipeline([('vect', CountVectorizer(max_features=10000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier())])
text_clf = text_clf.fit(train_texts, train_labels)
predicted = text_clf.predict(test_texts)
print np.mean(predicted == test_labels)

text_clf = Pipeline([('vect', CountVectorizer(max_features=10000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression())])
text_clf = text_clf.fit(train_texts, train_labels)
predicted = text_clf.predict(test_texts)
print np.mean(predicted == test_labels)

text_clf = Pipeline([('vect', TfidfVectorizer(max_features=10000)),
                      ('clf', MLPClassifier())])
text_clf = text_clf.fit(train_texts, train_labels)
predicted = text_clf.predict(test_texts)
print np.mean(predicted == test_labels)

text_clf = Pipeline([('vect', CountVectorizer(max_features=10000)),
                      ('clf', MLPClassifier())])
text_clf = text_clf.fit(train_texts, train_labels)
predicted = text_clf.predict(test_texts)
print np.mean(predicted == test_labels)
