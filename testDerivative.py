# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 18:56:30 2018

@author: Andrei ROG
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.decomposition import PCA
from random import seed
from random import random
from random import gauss
import random
import copy
from scipy import stats

def data_rand(N,M,sigma,Groups=2):
        
    
    # create the data container
    data_rand = []
    Labels = []
    # seed random number generator
    
    # generate random numbers between 0-1
    for _ in range(M):
        mean_random  = random.randint(50,150)#create one mean value
        v = []#create the sample points for each variable
        for k in range(N):
            v.append(gauss(mean_random, random.randint(sigma,2*sigma)))
        data_rand.append(v)
    for _ in range(N):
        Labels.append(random.randint(0,Groups-1))
    return data_rand,Labels
    
def add_signifficance(data,Labels,Groups,averageSig,sigma,sigvars):
    sig = []
    for j in Groups:
        if j>0:
            for v in sigvars:
                k = random.randint(averageSig-2*sigma,averageSig+2*sigma) + gauss(0, random.randint(sigma,2*sigma))
                sig.append(k)
                data[Labels==j,v] = data[Labels==j,v] + k
    return data,sig
    
        

def JSDe(X,Y,w,k):
    #project the data to k
    N,M = np.shape(X)
    T = np.repeat([k],N,0)
    xp = np.sort(np.sum(X*T,1)/np.linalg.norm(k)**2)
    j = 0
    JSDe = 0
    C = np.unique(Y)
    for c in C:
        Xc = X[Y==c,:]
        n,m = np.shape(Xc)
        T = np.repeat([k],n,0)        
        xc = np.sort(np.sum(Xc*T,1)/np.linalg.norm(k)**2)
        nc = np.shape(xc)[0]
        jsd = 0
        for i in np.arange(1,nc-1,1):
#            sx = np.min([xp[i]-xp[i-1],xp[i+1]-xp[i]])
#            rx = np.min(np.abs(xc[xc!=xp[i]]-xp[i]))
            sx = np.min(np.abs(xp[xp!=xc[i]]-xc[i]))
            rx = np.min(np.abs([xc[i]-xc[i-1],xc[i+1]-xc[i]]))
            if rx == 0 or sx == 0:                
                jsd += (0 + np.log2(N/(nc-1)))
            else:
                jsd += (np.log2(sx/rx) + np.log2(N/(nc-1)))
        JSDe += (w[j]/nc)*jsd
        j += 1
    return JSDe            
def JSD(X,Y,w,k,nr_pts,hist = False):            
    N,M = np.shape(X)
    T = np.repeat([k],N,0)
    xp = np.sort(np.sum(X*T,1)/np.linalg.norm(k)**2)
    
#    print(np.log2(Slack*nr_pts))
    pts = np.linspace(np.min(xp),np.max(xp),nr_pts)
    C = np.unique(Y)
    j = 0    
    Hc = 0
    jsd = 0
    fmix = np.zeros(np.shape(pts)[0])
#    plt.figure()
    if hist == False:
        for c in C:
            Xc = X[Y==c,:]
            n,m = np.shape(Xc)
            T = np.repeat([k],n,0)        
            xc = np.sort(np.sum(Xc*T,1)/np.linalg.norm(k)**2)
            KernelD = stats.gaussian_kde(xc,bw_method='scott')
            f = KernelD.evaluate(pts)
            fmix = fmix + w[c]*f
    #        plt.plot(pts,f)
            Fx = (f[0:-1] + 0.5*np.diff(f))
            Fx = Fx/np.sum(Fx)
    
            Hc = Hc - w[c]*(np.sum(Fx[Fx>0]*np.log2(Fx[Fx>0])))
    #        Hc = Hc - w[c]*np.trapz(f[f>0]*np.log2(f[f>0]),pts[f>0])                
            j+=1    
    #    plt.plot(pts,fmix)
        Fmix = (fmix[0:-1] + 0.5*np.diff(fmix))
        H = Fmix/np.sum(Fmix)    
        Hmix = -(np.sum(H[H>0]*np.log2(H[H>0])))
    else:
        for c in C:
            Xc = X[Y==c,:]
            n,m = np.shape(Xc)
            T = np.repeat([k],n,0)        
            xc = np.sort(np.sum(Xc*T,1)/np.linalg.norm(k)**2)            
            f,_ = np.histogram(xc,bins = nr_pts,range=[pts[0],pts[-1]])
            
    #        plt.plot(pts,f)            
            f = f/np.sum(f)
    
            Hc = Hc - w[c]*(np.sum(f[f>0]*np.log2(f[f>0])))
    #        Hc = Hc - w[c]*np.trapz(f[f>0]*np.log2(f[f>0]),pts[f>0])                
            j+=1    
    #    plt.plot(pts,fmix)
        n,m = np.shape(X)
        T = np.repeat([k],n,0)    
        Xc = np.sort(np.sum(X*T,1)/np.linalg.norm(k)**2)
        
        Fmix,_ = np.histogram(Xc,bins = nr_pts,range=[pts[0],pts[-1]])
        H = Fmix/np.sum(Fmix)    
        Hmix = -(np.sum(H[H>0]*np.log2(H[H>0])))
    jsd = Hmix - Hc
    if jsd<0:
        print('negative')
    return jsd    

def gradJSDe(X,Y,w,k):
    dG = None
    N,M = np.shape(X)
    T = np.repeat([k],N,0)
    xp = np.sort(np.sum(X*T,1)/np.linalg.norm(k)**2)
    
    dG = 0
    C = np.unique(Y)
    for j,c in enumerate(C):
        Xc = X[Y==c,:]
        n,m = np.shape(Xc)
        T = np.repeat([k],n,0)        
        xc = np.sort(np.sum(Xc*T,1)/np.linalg.norm(k)**2)
        nc = np.shape(xc)[0]
        jsd = 0
        for i in np.arange(1,nc-1,1):
#            sx = np.min([xp[i]-xp[i-1],xp[i+1]-xp[i]])
#            rx = np.min(np.abs(xc[xc!=xp[i]]-xp[i]))
            
            sx = np.min(np.abs(xp[xp!=xc[i]]-xc[i]))
            dvc = [xc[i]-xc[i-1],xc[i+1]-xc[i]]
            
            rx = np.min(np.abs(dvc))
            if np.abs(dvc[0])<np.abs(dvc[1]):
                id_xc = i-1
            else:
                id_xc = i+1
            id_xp = np.where(np.abs(xp[xp!=xc[i]]-xc[i]) == np.min(np.abs(xp[xp!=xc[i]]-xc[i])))
            redX = X[xp!=xc[i],:]
            red_xp = xp[xp!=xc[i]]
            aq = np.squeeze(redX[id_xp,:])
            ai = Xc[id_xc,:]
            aj = Xc[i]
            delta_xi = xc[i] - xc[id_xc]
            delta_xp = xc[i] - red_xp[id_xp]
            if rx == 0 or sx == 0:                
                jsd += 0
            else:
#                jsd += (xc[i]*ai + red_xp[id_xp]*aq)/sx**2 - (xc[i]*ai + xc[id_xc]*aj)/rx**2  
                jsd += ((np.abs(delta_xi)*delta_xp*[ai-aq] - np.abs(delta_xp)*delta_xi*[ai-aj])/(np.abs(delta_xp)*np.abs(delta_xi)))
        dG += (w[j]/nc/np.log(2))*jsd
    dG = np.squeeze(dG)    
    dx = dG[0]
    dy = dG[1]
     
    return dx,dy,dG
    
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    N = 200
    M = 2
    CLASSES = 2
    random.seed
    data_rand,Labels = data_rand(N,M,20,Groups=CLASSES)
    data_rand = np.array(data_rand, dtype='float32').T
    Labels = np.array(Labels,dtype='int32')    
    j = 0
    varSig = []
    while j< M:
        v = random.randint(0,M-1)
        if v not in varSig:
            varSig.append(v)
            j += 1   
    
    averageSig = 350
    sigma = 10
    data_sig,sig = add_signifficance(copy.deepcopy(data_rand),Labels,np.arange(0,CLASSES,1),averageSig,sigma,varSig)
    w = (1/CLASSES)*np.ones(CLASSES)
#    #######PCA####
#    pca = PCA(n_components=1)#create an instance of PCA
#    pca.fit(data_sig)
#    L_pca = pca.components_
#    Xp_train_pca = np.dot(np.dot(data_sig,L_pca.T),np.linalg.pinv(np.dot(L_pca,L_pca.T)))
    
    Span = np.arange(-1,1,0.01)   
    JS_es1 = np.zeros([np.shape(Span)[0],np.shape(Span)[0]])
    JS_es2 = copy.deepcopy(JS_es1)
    JS_es3 = copy.deepcopy(JS_es1)
    dX = np.zeros([np.shape(Span)[0],np.shape(Span)[0]])
    dY = np.zeros([np.shape(Span)[0],np.shape(Span)[0]])
    p =0
    
    for p,i in enumerate(Span):
        q = 0
        for q,j in enumerate(Span):
            k = np.array([i,j])/np.linalg.norm([i,j])
#            k =  [0.01*random.randint(1,100) for _ in range(M)]
#            k = k-np.mean(k)
           
#            JS_estim = JSDe(data_rand,Labels,w,k)    
#            JS_kdens = JSD(data_rand,Labels,w,k,100)
            dx,dy,_ = gradJSDe(data_sig,Labels,w,k)
            JS1 = JSDe(data_sig,Labels,w,k)
            JS2 = JSD(data_sig,Labels,w,k,100)
            JS3 = JSD(data_sig,Labels,w,k,100,hist=True)
            JS_es1[p,q] = JS1
            JS_es2[p,q] = JS2
            JS_es3[p,q] = JS3
            dX[p,q] = dx
            dY[p,q] = dy
            q = q+1
        
    X, Y = np.meshgrid(Span, Span) 
    fig = plt.figure()
    ax = fig.gca(projection='3d')      
    # Plot the surface.
    surf = ax.plot_surface(X, Y, JS_es1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title('Eucliden dist JSd')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')      
    # Plot the surface.
    surf = ax.plot_surface(X, Y, JS_es2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title('Gauss kernel JSd')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')      
    # Plot the surface.
    surf = ax.plot_surface(X, Y, JS_es3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title('Histogram JSd')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
#    np.delete(data_sig)
    # quiver from the smoothed version
    dy, dx = np.gradient(JS_es2, X[0,:], Y[:,0])
#    fig, ax = plt.subplots()
#    ax.quiver(X, Y, dx, dy, JS_es2)
#    ax.set(aspect=1, title='Quiver Plot of the kernel density')
#    plt.show()
    
    
    skip = (slice(None, None, 3), slice(None, None, 3))
    fig, ax = plt.subplots()
    ax.quiver(X[skip], Y[skip], dx[skip], dy[skip], JS_es2[skip])
    ax.set(aspect=1, title='Quiver Plot of the kernel')
    plt.show()
    
    fig, ax = plt.subplots()
    im = ax.imshow(JS_es2, extent=[X.min(), X.max(), Y.min(), Y.max()],origin='lower')
    ax.quiver(X[skip], Y[skip], dx[skip], dy[skip])
    
    fig.colorbar(im)
    ax.set(aspect=1, title='Quiver Plot of the kernel')
    plt.show()
    
    
    fig, ax = plt.subplots()    
    ax.streamplot(X, Y, dx, dy, color=JS_es2, density=0.5, cmap='gist_earth')    
    cont = ax.contour(X, Y, JS_es2, cmap='gist_earth')
    ax.clabel(cont)    
    ax.set(aspect=1, title='Streamplot with contours kernel density')
    plt.show()
    
    #theoretical gradient from the k-nn
#    fig, ax = plt.subplots()
#    ax.quiver(X, Y, dX, dY, JS_es1)
#    ax.set(aspect=1, title='Quiver Plot from the theoretical k-nn')
#    plt.show()
        
        
    skip = (slice(None, None, 3), slice(None, None, 3))
    fig, ax = plt.subplots()
    ax.quiver(X[skip], Y[skip], dX[skip], dY[skip], JS_es1[skip])
    ax.set(aspect=1, title='Quiver Plot from theoretical k-nn')
    plt.show()
        
    fig, ax = plt.subplots()
    im = ax.imshow(JS_es1, extent=[X.min(), X.max(), Y.min(), Y.max()],origin='lower')
    ax.quiver(X[skip], Y[skip], dX[skip], dY[skip])
        
    fig.colorbar(im)
    ax.set(aspect=1, title='Quiver Plot from theoretical k-nn')
    plt.show()        
        
    fig, ax = plt.subplots()        
    ax.streamplot(X, Y, dX, dY, color=JS_es1, density=0.5, cmap='gist_earth')        
    cont = ax.contour(X, Y, JS_es1, cmap='gist_earth')
    ax.clabel(cont)        
    ax.set(aspect=1, title='Streamplot with contours from theoretical k-nn')
    plt.show()
    
    #gradient from the k-nn surface
    dYe, dXe = np.gradient(JS_es1, X[0,:], Y[:,0])
#    fig, ax = plt.subplots()
#    ax.quiver(X, Y, dX, dY, JS_es1)
#    ax.set(aspect=1, title='Quiver Plot from k-nn surface')
#    plt.show()
        
        
    skip = (slice(None, None, 3), slice(None, None, 3))
    fig, ax = plt.subplots()
    ax.quiver(X[skip], Y[skip], dXe[skip], dYe[skip], JS_es1[skip])
    ax.set(aspect=1, title='Quiver Plot from k-nn surface')
    plt.show()
        
    fig, ax = plt.subplots()
    im = ax.imshow(JS_es1, extent=[X.min(), X.max(), Y.min(), Y.max()],origin='lower')
    ax.quiver(X[skip], Y[skip], dXe[skip], dYe[skip])
        
    fig.colorbar(im)
    ax.set(aspect=1, title='Quiver Plot')
    plt.show()
    ax.set(title='K-nn surface quiver')    
        
    fig, ax = plt.subplots()        
    ax.streamplot(X, Y, dXe, dYe, color=JS_es1, density=0.5, cmap='gist_earth')        
    cont = ax.contour(X, Y, JS_es1, cmap='gist_earth')
    ax.clabel(cont)        
    ax.set(aspect=1, title='Streamplot with contours from k-nn surface')
    plt.show()
    #gradient estimation from histogram entropy
    dYh, dXh = np.gradient(JS_es3, X[0,:], Y[:,0])
#    fig, ax = plt.subplots()
#    ax.quiver(X, Y, dXe, dYe, JS_es3)
#    ax.set(aspect=1, title='Quiver Plot')
#    plt.show()
        
        
    skip = (slice(None, None, 3), slice(None, None, 3))
    fig, ax = plt.subplots()
    ax.quiver(X[skip], Y[skip], dXh[skip], dYh[skip], JS_es3[skip])
    ax.set(aspect=1, title='Quiver Plot histogram entropy')
    plt.show()    
    fig, ax = plt.subplots()
    im = ax.imshow(JS_es3, extent=[X.min(), X.max(), Y.min(), Y.max()],origin='lower')
    ax.quiver(X[skip], Y[skip], dXh[skip], dYh[skip])
        
    fig.colorbar(im)
    ax.set(aspect=1, title='Quiver Plot histogram entropy')
    plt.show()        
        
    fig, ax = plt.subplots()        
    ax.streamplot(X, Y, dXh, dYh, color=JS_es3, density=0.5, cmap='gist_earth')