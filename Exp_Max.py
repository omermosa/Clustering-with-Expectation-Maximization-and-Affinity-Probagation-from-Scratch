# -*- coding: utf-8 -*-
"""
Created on Mon May 10 01:40:10 2020

@author: OmerHassan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("mnist_train.csv")
TruL=data['label'].values
data.drop(["label"],axis=1,inplace=True)
data=data.values
def Prob(X,Cov,mean,Prt,j,it):
    
    d=X.shape[1]
    m=X.shape[0]
    dt=(2*np.pi)*((np.linalg.det(Cov))**(-0.5))
    inv=np.linalg.inv(Cov)
    f2=X-mean
    pred=dt*np.exp(-0.5*np.einsum('ij, ij -> i', f2,(np.dot( inv , f2.T).T) ))
    Prt[:,j]=pred
    print("one cluster done")
    

def E_step(Pis,k,Prt,means,Covs,X,it):
    for j in range(k):
        Prob(X,Covs[j],means[j],Prt,j,it)
        
        Prt[:, j] = Pis[j] * Prt[:, j] #append all prob for all clusters
   
    Prt = (Prt.T / np.sum(Prt, axis = 1)).T # normalization step
    Sumps = np.sum(Prt, axis=0) # summation of pi after normalizaiton
    return (Sumps)
        

def M_step(Pis,k,Prt,means,Covs,Sumps,X):
    m=X.shape[0]
    
    for j in range(k):
        
        means[j] = 1.0 / Sumps[j] * np.sum(Prt[:, j] * X.T, axis = 1).T*10000 #update means
        mean_numj = np.matrix(X - means[j]) # to get covarience mat
        
        #cov=matrix of all sigmas (result is d by d mat.)
        Covs[j] = np.array(1 / Sumps[j] * np.dot(np.multiply(mean_numj.T,  Prt[:, j]),mean_numj)) # var=sigma (x-mean)^2*x*pi
        Pis[j] = 1.0 / m * Sumps[j] #update indv probabilities of each cluster = sum of pts for it / sum ofnum of totsl pts
        
        
        
    
def E_M(X,k,iterations):
    
    # initalize means and Cov, indv probailities, total probabilities
    m, d = X.shape      
    ind=np.random.randint(0,m,k)
    means = X[ind]  #choose K points from X as the starting means  
    Covs= [np.eye(d)] * k # we want k of n by n matrix (3d mat)
    Pis = [1.0/k] *k  #we want the indv probs to be for each cluster
    Prt = np.ones((m, k)) # want m-probabiliteis for each cluster (m*k matrix)
    #iterate over the two steps
    for it in range(iterations):
        print(it)
        Sumps=E_step(Pis,k,Prt,means,Covs,X,it)
        M_step(Pis,k,Prt,means,Covs,Sumps,X)
        
    return (means, Prt)



mus,P=E_M(data,10,300) # call with the num of clusters and iterations
    

Labels=[]

for i in range(P.shape[0]):
    Labels.append(np.argmax(P[i]))
print(TruL,Labels)

from sklearn.metrics import classification_report
print(classification_report(TruL, np.array(Labels)))

for mean in mus:
    plt.imshow(mean.reshape(28,28)) #reshape the pitcture
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


        
        
        
        
        
        
        

