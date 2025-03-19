import sys
# import words
# import we
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
import numpy as np
import re, sys
import random
import pickle as pkl
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import sklearn.feature_selection 
import csv
from collections import Counter
import json
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
import time
from sklearn.cluster import KMeans
import scipy.stats
from sklearn.model_selection import KFold

def corr(a,b):
    return np.corrcoef(a,b)[0,1]

def normalize(v):
    return v / np.linalg.norm(v)
    
def revsorted(*args, **kw_args):
    return sorted(*args, **kw_args, reverse=True)

def cross_val_corr_w_std(words, scores, E, n_reps=1, n_folds=10, clr = LinearRegression()):
    MSEs = []
    correlations = []
    for rep in range(n_reps):
        kf = KFold(n_splits=n_folds, random_state=rep, shuffle=True)    
        word_predict = {}
        tot_error = 0
        for train,test in kf.split(words):
            train_words = [words[i] for i in train]
            test_words = [words[i] for i in test]
#             print(len(train_words), len(test_words))
            X_train, X_test = [[E[w] for w in _words] for _words in (train_words, test_words)]
            y_train, y_test = [[scores[w] for w in _words] for _words in (train_words, test_words)]
            clr.fit(X_train, y_train)
            y_hat = clr.predict(X_test)
            word_predict.update({w: _y_hat for w, _y_hat in zip(test_words, y_hat)})
            tot_error += np.sum(np.square(y_hat-y_test))
        MSEs.append(tot_error/len(words))
        correlations.append(corr([word_predict[w] for w in words], [scores[w] for w in words]))
    return np.mean(correlations)#, std(correlations)*1.96

def cross_val_corr(words, scores, E, n_reps=1, n_folds=10, clr = LinearRegression()):
    return cross_val_corr_w_std(words, scores, n_reps, n_folds, clr, E)[0]


def scatter_embedding(EH_scores, E, n_reps=1, fig_gen=False, save_filename=None, size=0.5):
    clr = LinearRegression()
    clr.fit([E[w] for w in EH_scores], [EH_scores[w] for w in EH_scores]) 
    test_words = [w for w in EH_scores]
    y_test = [EH_scores[w] for w in test_words]
    y_hat = clr.predict([E[w] for w in test_words])
    if fig_gen:
        figure(figsize=(4,3))
        scatter(y_hat, y_test, s=size)
        plt.xlabel('predicted humor rating')
        plt.ylabel('humor rating from EH')
        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename, dpi=300)
    return corr(y_test, y_hat), cross_val_corr_w_std([w for w in EH_scores], EH_scores, E, n_reps=n_reps)

def get_cockamamie():
    cockamamie_gobbledegook_us_data = []
    with open("cockamamie_gobbledegook_us_data.json", "r") as f:
        cockamamie_gobbledegook_us_data = json.load(f)
    #keys = cockamamie_gobbledegook_us_data.keys()
    #print(cockamamie_gobbledegook_us_data["word_features"])
    return cockamamie_gobbledegook_us_data

def get_EH():
    with open("humor_dataset.csv", "r") as f:
        foo = [line.strip().split(",") for line in f.readlines()]
        headings = foo[0][1:]
        others = {line[0]: {feat: float(v) for v, feat in zip(line[1:], headings)} for line in foo[1:]}
    # E_other.filter_words(lambda w: w in others)
    EH_scores = {w: others[w]["mean"] for w in others}

    #print(len(EH_scores))
    # print(len(other_scores), "words in the embedding")
    return EH_scores
		
def print_missing(E, words_of_interest): #= [("EH", list(EH_scores)), ("120k", wordss[0])]):
    for name, words in words_of_interest:
        missing = []
        for w in words:
            try:
                E[w.replace("_", " ")]
            except:
                missing.append(w)
        print(f"{len(missing):,} missing words ({len(missing)/len(words):.1%}).")
        if missing:
            print(", ".join(missing[:10]) + ("" if len(missing)<10 else ", ..."))

def vec_file2dict(vec_filename, pkl_filename, words_of_interest):
    from gensim.models import KeyedVectors

    e = KeyedVectors.load_word2vec_format(vec_filename)
    def get_word(w):
        try:
            return normalize(e[w.replace("_", " ")])
        except:
            return 0.0*e["cat"]

    e_dict = {w: get_word(w) for name, words in words_of_interest for w in words}
    with open(pkl_filename, "wb") as f:
        pkl.dump(e_dict, f)

    print(f"Loaded '{vec_filename}' and wrote {len([v for v in e_dict.values() if not np.allclose(v, 0)]):,} nonzero vectors to '{pkl_filename}'")
        
    print_missing(e_dict, words_of_interest)
