import numpy as np 
from .pse import *

class DiscreteLabelSpace:
    def __init__(self,lst_unq_labels):
        self.lst_unq_labels = lst_unq_labels
        self.k = len(lst_unq_labels)
        self.idx2lbl = lst_unq_labels 
        self.lbl2idx = dict(zip(lst_unq_labels,range(self.k)))
        self.embeddings = {"1-hot":None, "pse":None}
        self.distances = {"zero-one":None}
    
    def dist_zero_one(self,i,j):
        if(i==j):
            return 0
        else:
            return 1
    
    def compute_distances(self,distance_type):
        if(distance_type in self.distances and self.distances[distance_type] is not None):
            return self.distances[distance_type]
        
        D = np.ones((self.k,self.k))
        if(distance_type=='zero-one'):
            for i in range(self.k):
                D[i][i] = 0
        self.distances[distance_type] = D 
        return D 
        
    def compute_embeddings(self,embedding_type,distance_type,dim):
        D = self.compute_distances(distance_type) 
        k = self.k 

        if(embedding_type=='1-hot'):
            Yspace_emb = np.zeros((k,k))
            Yspace_emb[:,0:k] =  np.eye(k)
            
            self.embeddings['1-hot'] = Yspace_emb

        elif(embedding_type=='pse'):
            D = np.array(D)
            Yspace_emb, tk = pseudo_embedding(D, dim)
            
            print(Yspace_emb)
            d_ = len(Yspace_emb[0])
            if(dim<tk):
                Yspace_emb_pos = Yspace_emb[:, :dim]
                d_neg = 0
            else:
                Yspace_emb_pos = Yspace_emb[:, :tk]
                Yspace_emb_neg = Yspace_emb[:, tk: min(dim,d_)]  # NOTE this could be 0-dimensional
                d_neg =  len(Yspace_emb_neg[0])

            print(tk)
            self.embeddings['pse']= {'pos':Yspace_emb_pos}
            
            if(d_neg>0):
                self.embeddings['pse']['neg'] = Yspace_emb_neg
