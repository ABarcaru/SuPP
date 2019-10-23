# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:45:30 2018

@author: Andrei ROG
"""
from scipy import optimize
from scipy import stats
import numpy as np

class SuPP:       
    def __init__(self,k = 1,options=None):
        if options is None:
            options  = {'un_classes':0,
             'nr_classes':0,#comes from the data
             'N':0,#comes from the dava
             'M':0,#comes from the data
             'n':100,         
             'orth_weight':64,#2*np.log2(3)
             'discrete_entropy':False,
             'bw_method':'scott',
             'optimization':{'minmethod' : 'COBYLA','Temp':1000,'disp':True,'niter':200,}
             }                                  
        self.k = k#number of latent components
        self.opts = options
        self.minmethod = self.opts['optimization']['minmethod']
        if self.opts['bw_method'] is None:
            self.opts['bw_method'] = 'scott'
        if self.opts['optimization']['display'] is None:
            self.opts['optimization']['display'] = False
        if self.opts['optimization']['niter'] is None:
            self.opts['optimization']['niter'] = 100                    
        self.g = []
    def pursuit(self,X,y,w,X0=None):
        k = self.k        
        bnds = ()
        N,M = np.shape(X)
        CLASSES = np.shape(np.unique(y))
        if self.opts['M'] == 0 and M>0:
            self.opts['M'] = M
        if self.opts['N'] == 0 and N>0:
            self.opts['N'] = N
        if self.opts['nr_classes'] == 0 and CLASSES>0:
            self.opts['nr_classes'] = CLASSES                          
        for j in range(self.opts['M']):
            bnds = (bnds) + ((-1,1),)
        kwarg = {'method':self.minmethod,'bounds':bnds}
        self.g = np.zeros([k,self.opts['M']])
        for j in range(k):
            self.opts['j'] = j
            if X0 is None:
                x0 = np.random.rand(1,self.opts['M'])
            else:
                x0 = X0[j,:]
            x0 = x0/np.linalg.norm(x0)
            res = optimize.basinhopping(lambda x: self.objectFun(x,y,X,w,self.opts), x0, T = self.opts['optimization']['Temp'], disp=self.opts['optimization']['display'], minimizer_kwargs = kwarg,niter=self.opts['optimization']['niter'])
            if self.opts['optimization']['display'] == True:
                print(res)            
            self.g[j,:] = res.x/np.linalg.norm(res.x)            
        self.scores = self.getScores(X)
        return self.g
    
    def objectFun(self,x,Y,X,w,opts):
        k = self.g       
        n = opts['j']                
        k[n,:] = x/np.linalg.norm(x)#append the new coordinate                     
        J = 0    
        O = 0#initializing orthogonality
        Xk = self.projectVect(X,k[n,:])        
        pts = np.linspace(np.nanmin(Xk),np.nanmax(Xk),opts['n'])
        Hx = 0
        Mix = np.zeros([1,opts['n']])
        bw = opts['bw_method']
        if self.opts['discrete_entropy'] == True:
            for ids in range(opts['nr_classes']):            
                f,_ = np.histogram(Xk[Y == opts['un_classes'][ids]],bins = opts['n'],range=[pts[0],pts[-1]])
                f = f/np.sum(f)                
                Hx = Hx - w[ids]*np.nansum(f[f>0]*np.log2(f[f>0]))            
                Mix = Mix + w[ids]*f
            Mix,kk = np.histogram(Xk,bins = opts['n'],range=[pts[0],pts[-1]])
            Mix = Mix/np.sum(Mix)
            Hmix = -np.sum(Mix[Mix>0]*np.log2(Mix[Mix>0]))
        else:
            for ids in range(opts['nr_classes']):
                xk = Xk[Y == opts['un_classes'][ids]]                
                KernelD = stats.gaussian_kde(xk[~np.isnan(xk)],bw_method = bw)
                f = KernelD.evaluate(pts)#
#                making discretization of KDS
                Mix = Mix + w[ids]*f
                Fx = f#(f[0:-1] + 0.5*np.diff(f))
                Fx = Fx/np.sum(Fx)#               
                Hx = Hx - w[ids]*(np.sum(Fx[Fx>0]*np.log2(Fx[Fx>0])))
                #
            Fmix = Mix#(Mix[0,0:-1] + 0.5*np.diff(Mix))
            Fmix = Fmix/np.sum(Fmix)                   
            Hmix = -(np.sum(Fmix[Fmix>0]*np.log2(Fmix[Fmix>0])))        
        J = Hmix - Hx             
        if n>0:
            D = n+1
            u = np.ones([D,D],dtype=bool)
            e = np.eye(D,dtype=bool)            
            O = (2/(D*(D-1)))*np.sum(np.abs(np.dot(k[0:D,:],k[0:D,:].T)[np.triu((u!=e))]))
            Cost = (1-J/self.opts['epsilon'])**2 + self.opts['orth_weight']*O**2#  
        else:
            Cost = (1-J/self.opts['epsilon'])**2
        return Cost
    def projectVect(self,X,g):
        N = X.shape[0]
        T = np.repeat([g],N,0)
        return np.nansum(X*T,1)/np.linalg.norm(g)**2
    def getScores(self,X,g = None):
        if g is None:
            g = self.g
        n = X.shape[0]
        k = g.shape[0]
        Scores = []
        for i in range(k):
            Scores.append(self.projectVect(X,g[i,:]))
        if g is None:
            self.scores = np.array(Scores).T
        return np.array(Scores).T
    def gramSchmidt(self,g, row_vecs=True, norm = True):
        if not row_vecs:
            g = g.T
        y = g[0:1,:].copy()
        for i in range(1, g.shape[0]):
            proj = np.diag((g[i,:].dot(y.T)/np.linalg.norm(y,axis=1)**2).flat).dot(y)
            y = np.vstack((y, g[i,:] - proj.sum(0)))
        if norm:
            y = np.diag(1/np.linalg.norm(y,axis=1)).dot(y)
        if row_vecs:
            return y
        else:
            return y.T
    def relativeGroupDist(self,Y,X=None):
        if self.scores is None and X is not None:
            Scores = self.getScores(X)            
        elif self.scores is None and X is None:
            print("Error! the Scores are not set and the data was not passed.\nPlease Specify the data X or calculate the scores first using getScores(Data) method")            
        elif X is not None and self.Scores is not None:
            print("Warning! Data was passed with the previously calculated attribute Scores!\nRecalculating Scores")
            Scores = self.getScores(X)
        else:
            Scores = self.scores
        RelGD = []
        for i in range(Scores.shape[1]):
            Quant = []
            D = 0            
            for c in range(self.opts['nr_classes']):                
                Quant.append({'5p':np.nanquantile(Scores[Y == self.opts['un_classes'][c],i],0.05),
                              '50p':np.nanquantile(Scores[Y == self.opts['un_classes'][c],i],0.5),
                              '95p':np.nanquantile(Scores[Y == self.opts['un_classes'][c],i],0.95)})
            MinEntry = min(Quant,key = lambda x: x['5p'])
            MaxEntry = max(Quant,key = lambda x: x['5p'])
            for item in Quant:
                if item == MinEntry:
                    D += np.abs(item['95p'] - item['50p'])
                elif item == MaxEntry:
                    D += np.abs(item['50p'] - item['5p'])
                else:
                    D += np.abs(item['95p'] - item['5p'])
            RelGD.append((MaxEntry['50p']-MinEntry['50p'])/D)
        return RelGD
                        
            
            
            
            
        
    
           
    
        
            
                
                
        
        
        