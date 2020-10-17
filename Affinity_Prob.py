# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:49:43 2020

@author: OmerMoussa
"""

# this code implements affinity probagation 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data=pd.read_csv("mnist_test.csv")
TruL=data['label'].values
data.drop(["label"],axis=1,inplace=True)
data=data.values


def measure_sim(Xi,Xk):
    return -1*((Xi - Xk)**2).sum() 

def SimilarityMatrix(X ):
    m=X.shape[0]
    SimMat=np.ones((m,m))
    for i in range(m):
        for j in range(m):
            SimMat[i, j] = measure_sim(X[i], X[j])
    return SimMat

def Responsibility(R,X,S,A):
    m=X.shape[0]
    for i in range(m):
        for j in range(m):
            a = S[i, :] + A[i, :] # availability of A(i,j) + Similariry (i,j) (get the vector then get the max)
            a[j],a[i] = -np.inf, -np.inf # small value initialization
            R[i, j] = 0.8*R[i,j]+0.2*(S[i, j] - np.max(a)) # R(i,j)=(S(i,j)-max(s(i,j)+a(i,j)) )


def Availability(R,X,S,A):
    m=X.shape[0]
    for i in range(m):
        for j in range(m):
            r = np.array(R[:, j]) # jth column
            
            if i != j: # not self avilability
                r[i],r[j] = -np.inf,-np.inf
                r[r < 0] = 0 # if less than zero-> zero
                A[i, j] = 0.8*A[i,j]+0.2*min(0, R[j, j] + r.sum()) # Aij=min(0,rjj+sum(max(0,rij)))
            # The diagonal
            else: #self availability
                r[j] = -np.inf
                r[r < 0] = 0 
                A[j, j] = 0.8*A[i,j]+0.2*r.sum() #ajj=max(0,rij)

def Affinity_Prob(X, iterations, S):
    m=X.shape[0]
    #initialize
    R=np.zeros((m,m))
    A=np.zeros((m,m))
    
   
    Examplers=np.array([])
    for it in range(iterations):
        #updata A and R
        print(it)
        Responsibility(R,X,S,A)
        Availability(R,X,S,A)
        Examplers = np.unique(np.argmax( A + R, axis=1)) #choose examplers
    return Examplers
            
# Change self_similarity
S=SimilarityMatrix(data)

self_similarity = np.median(S)
np.fill_diagonal(S, self_similarity) #self similarity
Exs=Affinity_Prob(data,100,S)
   
for ex in Exs:
    plt.imshow(data[ex,:].reshape(28,28))
    plt.show()
