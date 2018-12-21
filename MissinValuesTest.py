# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 23:48:05 2018

@author: Andrei ROG
"""
import numpy as np
import random
import copy
import SuPP as supp
#X_train = np.load('X_train_last.npy')
#y_train = np.load('y_train_last.npy')
#X_test = np.load('X_test_last.npy')
#y_test = np.load('y_test_last.npy')
N,M = np.shape(X_train)
n = N*M
TEST_SET_FRACTION = 0.5
COMPONENTS = 2
CLASSES = 2
epsilon = np.log2(CLASSES)
w = 1/CLASSES * np.ones(CLASSES)
options  = {'un_classes':np.unique(y_train),
            'nr_classes':CLASSES,
            'N':N,#comes from the set of predictors
            'M':M,#comes from the set of predictors
            'n':100,
            'orth_weight':1,
            'discrete_entropy':False,
            'epsilon':epsilon,
            'bw_method':'scott',
            'optimization':{'minmethod' : 'COBYLA','Temp':10000,'display':True,'niter':100}}
#run the SuPP without the NaNs
C_supp = supp.SuPP(k=COMPONENTS,options=options)#create an instance of the class SuPP
L_supp = C_supp.pursuit(X_train,y_train,w)
Xp_test = C_supp.getScores(X_test)
#create the random id which will be switched to nan. maximum of 0.3*NxM
#maximum missing values will be 30% of the datapoints
L_array = []
X_array = []
L_array.append(L_supp)
X_array.append(Xp_test)#first entry in the array is the 0 rate of nan, i.e. original data
K = 0
randVals = []
while K<int(0.3*n):
    v = [random.randint(0,N-1),random.randint(0,M-1)]
    if v not in randVals:
        randVals.append(v)
        K+=1
randXNan = [x[1] for x in randVals]
randYNan = [x[0] for x in randVals]

Rates = np.array([0.01,0.05,0.1,0.3])*0.3

for r in Rates:
    X_trainNan = copy.deepcopy(X_train)
    X_trainNan[randYNan[0:int(r*int(0.3*n))],randXNan[0:int(r*int(0.3*n))]] = np.nan
    #run supp
    C_supp = supp.SuPP(k=COMPONENTS,options=options)#create an instance of the class SuPP
    L_supp = C_supp.pursuit(X_trainNan,y_train,w)
    #project the data
    Xp_test = C_supp.getScores(X_test)
    L_array.append(L_supp)
    #add to an array
    X_array.append(Xp_test)
    
    
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"    
    