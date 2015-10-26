import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stw import SupervisedTermWeightingWTransformer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, f1_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Construct term count matrix for train and test datasets
vectorizer = CountVectorizer(input='filename')

train_x = vectorizer.fit_transform(train['filename'])
train_y = train['target']

test_x = vectorizer.transform(test['filename'])
test_y = test['target']


# Use SVM as classifier

clf = LinearSVC()


# tf-idf unsupervised term weighting

transformer = TfidfTransformer()

train_x_t = transformer.fit_transform(train_x,train_y)
test_x_t  = transformer.transform(test_x)


# Train classifier and make predictions

clf.fit(train_x_t,train_y)
pred = clf.predict(test_x_t)


# Assess performance

print 'tf-idf scheme: accuracy = %0.2f, recall = %0.2f, f1 score = %0.2f' % \
(accuracy_score(test_y,pred), recall_score(test_y,pred), f1_score(test_y,pred))


# Supervised term weighting schemes

for scheme in ['tfchi2','tfig','tfgr','tfor','tfrf']:

    transformer = SupervisedTermWeightingWTransformer(scheme=scheme)

    train_x_t = transformer.fit_transform(train_x,train_y)
    test_x_t  = transformer.transform(test_x)


    # Train classifier and make predictions
    clf.fit(train_x_t,train_y)
    pred = clf.predict(test_x_t)


    # Assess performance
    print '%s scheme: accuracy = %0.2f, recall = %0.2f, f1 score = %0.2f' % \
    (scheme, accuracy_score(test_y,pred), recall_score(test_y,pred), f1_score(test_y,pred))
