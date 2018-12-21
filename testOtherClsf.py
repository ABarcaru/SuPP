# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:14:38 2018

@author: Andrei ROG
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#classification report
from sklearn.metrics import accuracy_score
def testOther(X_train,y_train,X_test,y_test,classifier,options):
    CLASSES = options['nr_classes']
    N = options['N']#comes from the set of predictors
    M = options['M']#comes from the set of predictors
    gamma = options['svm_gamma']
    C = options['svm_C']
    COMPONENTS = options['nr_components']
    if options['display_result']==True:
        import matplotlib.pyplot as plt
    results = {'pls':None,'pca':None,'lda':None,'svm':None,'gnb':None}
    if classifier['pls-da'] == True:
        pls2 = PLSRegression(n_components=COMPONENTS)
        q = np.unique(y_train)
        y_pred_pls = np.zeros(np.shape(y_test))
        YDA_train = np.zeros([N,CLASSES])
        for j in q:
            if 0 in q:
                YDA_train[np.where(y_train==j),j]=1
            else:                   
                YDA_train[np.where(y_train==j),j-1]=1 
        pls2.fit(X_train,YDA_train)
        YDA_pred = np.round(pls2.predict(X_test))
        for j in range(CLASSES):
            y_pred_pls[np.where(YDA_pred[:,j]==1)] = q[j]
        YDA_test = np.zeros([np.shape(y_test)[0],CLASSES])
        for j in q:
            if 0 in q:
                YDA_test[np.where(y_test==j),j]=1
            else:
                YDA_test[np.where(y_test==j),j-1]=1
        Xp_scores_test,Yp_scores_test = pls2.transform(X_test, Y=YDA_test, copy=True)
        results['pls'] = accuracy_score(y_test, y_pred_pls)
        if options['display_result']==True:            
            plt.figure()
            plt.scatter(pls2.x_scores_[:,0],pls2.x_scores_[:,1],c=y_train,marker='o',cmap='RdBu')            
            plt.scatter(Xp_scores_test[:,0],Xp_scores_test[:,1],c=y_test,marker='+',cmap='RdBu')
            plt.title('PLS-DA space')
    if classifier['pca'] == True:
        results['pca'] = [0,0]
        pca = PCA(n_components=COMPONENTS)#create an instance of PCA
        pca.fit(X_train)
        L_pca = pca.components_
        Xp_train_pca = np.dot(np.dot(X_train,L_pca.T),np.linalg.pinv(np.dot(L_pca,L_pca.T)))
        clf = GaussianNB()
        clf.fit(Xp_train_pca, y_train)
        Xp_test_pca = np.dot(np.dot(X_test,L_pca.T),np.linalg.pinv(np.dot(L_pca,L_pca.T)))
        y_pred_pca = clf.predict(Xp_test_pca)
        results['pca'][0] = accuracy_score(y_test, y_pred_pca)
        lin_clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
        lin_clf.fit(Xp_train_pca, y_train)    
        y_pred = lin_clf.predict(Xp_test_pca)        
        results['pca'][1] = accuracy_score(y_test, y_pred)
        if options['display_result']==True:            
            plt.figure()            
            plt.scatter(Xp_train_pca[:,0],Xp_train_pca[:,1],c=y_train,marker='o',cmap='RdBu')
            plt.scatter(Xp_test_pca[:,0],Xp_test_pca[:,1],c=y_test,marker='+',cmap='RdBu')
            plt.title('PCA space')
    if classifier['lda'] == True:
        lda = LinearDiscriminantAnalysis(n_components=COMPONENTS)#(solver='eigen',shrinkage='auto', n_components=COMPONENTS)
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        results['lda'] = accuracy_score(y_test, y_pred)
    if classifier['svm'] == True:
        lin_clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
        lin_clf.fit(X_train, y_train)    
        y_pred = lin_clf.predict(X_test)        
        results['svm'] = accuracy_score(y_test, y_pred)
    if classifier['gnb'] == True:
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results['gnb'] = accuracy_score(y_test, y_pred)
    return results
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
#    result = testOther(x_train_s,y_train[0:n],X_test,y_test,classifier=Classifiers,options=Opt)