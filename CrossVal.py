# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:04:53 2018

@author: Andrei ROG
"""

import SuPP as supp
import PLS as pls
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#algebra and statistics tools
from sklearn import preprocessing
import numpy as np
#data sets
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_wine,load_iris
#classifiers
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
#classification report
from sklearn.metrics import accuracy_score

import multiprocessing.pool as mp

def get_numbers(RS,DS,TSF,CI,CP,CL):
    RANDOM_STATE =RS#41
    DATA_SET = DS
    TEST_SET_FRACTION = TSF
    CROSS_ITERATIONS = CI 
    
    if DATA_SET == 'wine':
        features, target = load_wine(return_X_y=True)
    elif DATA_SET == 'iris':
        features, target = load_iris(return_X_y=True)
    
    i = 0
    O = []
    for TEST_SET_FRACTION in TSF:       
        X_train, X_test, y_train, y_test = train_test_split(preprocessing.scale(features), target,
                                                    test_size=TEST_SET_FRACTION,
                                                    random_state=int(RANDOM_STATE[i]))
        N,M = np.shape(X_train)
        i = i+1
        O.append(N)
    return O  
        

def cross_val_supp(RS,DS,TSF,CI,CP,CL,options):
    RANDOM_STATE =RS#41    
    TEST_SET_FRACTION = TSF
    CROSS_ITERATIONS = CI
    COMPONENTS = CP
    CLASSES = CL
    features = DS['features']
    target = DS['target']
    w = ((1/CLASSES)*np.ones([CLASSES,])).tolist()    
    acc_svm_supp = np.zeros(CROSS_ITERATIONS)    
    acc_svm_pca = np.zeros(CROSS_ITERATIONS)
    acc_gnb_supp = np.zeros(CROSS_ITERATIONS)    
    acc_gnb_pca = np.zeros(CROSS_ITERATIONS)
    acc_clf_pls = np.zeros(CROSS_ITERATIONS)
    acc_clf_lda = np.zeros(CROSS_ITERATIONS)
#    acc_clf_svm = np.zeros(CROSS_ITERATIONS)
#    acc_clf_gnb = np.zeros(CROSS_ITERATIONS)
    for i in range(CROSS_ITERATIONS):
        X_train, X_test, y_train, y_test = train_test_split(preprocessing.scale(features), target,
                                                    test_size=TEST_SET_FRACTION,
                                                    random_state=int(RANDOM_STATE[i]))
        N,M = np.shape(X_train)
        options['un_classes']=np.unique(y_train)
        options['nr_classes']=CLASSES
        options['N']=N#comes from the set of predictors
        options['M']=M#comes from the set of predictors
        options['epsilon'] = np.log2(CLASSES)
        C_supp = supp.SuPP(k=COMPONENTS,options=options)#create an instance of the class SuPP
        L_supp = C_supp.pursuit(X_train,y_train,w)
        
        Xp_train = C_supp.getScores(X_train)
        Xp_test = C_supp.getScores(X_test)
        #predicting with linear SVM
        lin_clf = svm.SVC(kernel='rbf', gamma=0.1, C=1)
        lin_clf.fit(Xp_train, y_train)    
        y_pred = lin_clf.predict(Xp_test)        
        acc_svm_supp[i] = accuracy_score(y_test, y_pred)
        clf = GaussianNB()
        clf.fit(Xp_train, y_train)
        y_pred = clf.predict(Xp_test)
        acc_gnb_supp[i] = accuracy_score(y_test, y_pred)
        pls2 = PLSRegression(n_components=COMPONENTS)
        q = np.unique(y_test)
        YDA_train = np.zeros([N,CLASSES])
        for j in q:
            if 0 in q:
                YDA_train[np.where(y_train==j),j]=1
            else:
                YDA_train[np.where(y_train==j),j-1]=1 
        pls2.fit(X_train,YDA_train)
        YDA_pred = np.round(pls2.predict(X_test))
        for j in range(CLASSES):
            y_pred[np.where(YDA_pred[:,j]==1)] = q[j]
        acc_clf_pls[i] = accuracy_score(y_test, y_pred)
        pca = PCA(n_components=COMPONENTS)#create an instance of PCA
        pca.fit(X_train)
        L_pca = pca.components_
        Xp_train_pca = np.dot(np.dot(X_train,L_pca.T),np.linalg.pinv(np.dot(L_pca,L_pca.T)))
        clf.fit(Xp_train_pca, y_train)
        Xp_test_pca = np.dot(np.dot(X_test,L_pca.T),np.linalg.pinv(np.dot(L_pca,L_pca.T)))
        y_pred_pca = clf.predict(Xp_test_pca)
        acc_gnb_pca[i] = accuracy_score(y_test, y_pred_pca)
        lin_clf = svm.SVC(kernel='rbf', gamma=0.1, C=1)
        lin_clf.fit(Xp_train_pca, y_train)    
        y_pred = lin_clf.predict(Xp_test_pca)        
        acc_svm_pca[i] = accuracy_score(y_test, y_pred)
        lda = LinearDiscriminantAnalysis(n_components=COMPONENTS)#(solver='eigen',shrinkage='auto', n_components=COMPONENTS)
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        acc_clf_lda[i] = accuracy_score(y_test, y_pred)
        #----pure svm and pure gnb
#        lin_clf = svm.SVC(kernel='rbf', gamma=0.1, C=5.5)
#        lin_clf.fit(X_train, y_train)    
#        y_pred = lin_clf.predict(X_test)        
#        acc_svm[i] = accuracy_score(y_test, y_pred)
#        clf = GaussianNB()
#        clf.fit(X_train, y_train)
#        y_pred = clf.predict(X_test)
#        acc_gnb[i] = accuracy_score(y_test, y_pred)
        print('#### --- Executed iteration ',i)
    return acc_svm_supp, acc_svm_pca, acc_gnb_supp, acc_gnb_pca, acc_clf_pls, acc_clf_lda
    
RANDOM_STATE = np.linspace(25,55,30)#41
DATA_SET = 'wine'
TEST_SET_FRACTION = 0.35
CROSS_ITERATIONS = 30
COMPONENTS = 3
CLASSES = 2
Fractions = np.linspace(0.16,0.72,5)#np.linspace(0.1,0.85,10)
w = 1/CLASSES * np.ones(CLASSES)
options  = {'un_classes':0,
         'nr_classes':CLASSES,
         'N':0,#comes from the set of predictors
         'M':0,#comes from the set of predictors
         'n':100,         
         'orth_weight':CLASSES**4,#2*np.log2(3)
         'discrete_entropy':False,
         'bw_method':'scott',
         'optimization':{'minmethod' : 'COBYLA','Temp':100,'display':True,'niter':100}
         }
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    if DATA_SET == 'wine':
        features, target = load_wine(return_X_y=True)
    elif DATA_SET == 'iris':
        features, target = load_iris(return_X_y=True)
    elif DATA_SET == 'carcinoma':
        features =np.load('X_carcinoma.npy').tolist()
        target = np.load('Y_carcinoma.npy').tolist()
    DS = {'features':features,'target':target}
    pool = mp.Pool(processes=6)
    results = [pool.apply_async(cross_val_supp, args=(RANDOM_STATE,DS,FRC,CROSS_ITERATIONS,COMPONENTS,CLASSES,options)) for FRC in Fractions]
#    results = [cross_val_supp(RANDOM_STATE,DS,FRC,CROSS_ITERATIONS,COMPONENTS,CLASSES,options) for FRC in Fractions]
    ListACC = [p.get() for p in results]
    np.save('wine_comp3.npy',np.array(ListACC))

