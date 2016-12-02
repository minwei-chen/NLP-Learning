# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:26:44 2016

@author: Clavicus
"""
from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import re
from sklearn import metrics



df = pd.read_csv(r'twitter-sanders-apple2.csv')
le = preprocessing.LabelEncoder()
df.label=le.fit_transform(df.label)

word_set = set()

for item in df.text:
    word_list = ' '.join(re.split(r"[^a-zA-Z]", item.lower())).split()
    for word in word_list:
        word_set.add(word)

word_set -= {"to", "a", "for", "the", "an", "on"}

word_idx = {}

size = 0

for word in word_set:
    word_idx[word] = size
    size += 1
    
def get_features(item):
    word_list = ' '.join(re.split(r"[^a-zA-Z]", item.lower())).split()
    base = len(word_list)
    rs = [0 for i in range(size+1)]
    #rs.append(base)
    for word in word_list:
        rs[word_idx.get(word, size)] += 1

    #rs = [(rs[i])/base for i in range(size)]
    return rs

features=[]
for item in df.text:
    features.append(get_features(item))

labels=df.label
"""
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
features= vectorizer.fit_transform(df.text)
"""
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(features)

"""
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
tfidf=vectorizer.fit_transform(df.text)
"""

X_train, X_test, y_train, y_test = train_test_split(tfidf,labels, test_size=0.2, random_state=50)

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB() 
bnb.fit(X_train, y_train) 
y_pred=bnb.predict(X_test) 
print("Naive_Bayes: "+str(np.mean(y_pred == y_test)))
print(metrics.classification_report(y_test,y_pred))

from sklearn.linear_model import Perceptron
pcp = Perceptron(penalty='l2', n_iter=20) 
pcp.fit(X_train, y_train) 
y_pred=pcp.predict(X_test) 
print("Perceptron: "+str(np.mean(y_pred == y_test)))
print(metrics.classification_report(y_test,y_pred)) 

from sklearn.linear_model import SGDClassifier 
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X_train, y_train) 
y_pred=clf.predict(X_test) 
print("SGD: "+str(np.mean(y_pred == y_test)))
print(metrics.classification_report(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test) 
print("Random_Forest: "+str(np.mean(y_pred == y_test)))
print(metrics.classification_report(y_test,y_pred))

from sklearn.neighbors.nearest_centroid import NearestCentroid 
clf = NearestCentroid()
clf.fit(X_train, y_train)
NearestCentroid(metric='euclidean', shrink_threshold=None)
y_pred=clf.predict(X_test)
print("Nearest_Centroid: "+str(np.mean(y_pred == y_test)))
print(metrics.classification_report(y_test,y_pred))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=10)

bdt.fit(X_train, y_train)
y_pred=bdt.predict(X_test) 
print("SKTree: "+str(np.mean(y_pred == y_test)))
print(metrics.classification_report(y_test,y_pred))