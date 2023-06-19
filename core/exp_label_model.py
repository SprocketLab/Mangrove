import numpy as np 

class ExponentialLabelModel:
    
    def __init__(self):
        pass 
    
    def get_probability_table_true(self,space, theta, D):
        """ """
        self.theta = theta
        self.space = space 
     
        potentials = np.zeros((theta.shape[0], len(space), len(space)))
        for i in range(len(space)):  # All possible rankings
            for j in range(len(space)):
                lam = space[i]
                y = space[j]
                # Compute un-normalized potentials
                d = D[lam][y] #dist_fun(lam, y)
                potentials[:, i, j] = np.exp(-theta * d**2)

        partition = potentials.sum(axis=1, keepdims=True)
        probabilities = potentials / partition
        # Indexing should be (LF indices, Center indices, indices of specific LF values)
        probabilities = probabilities.transpose((0, 2, 1))
        self.probabilities = probabilities
        return probabilities
    
    def draw_samples(self,y_centers):
        L = []
        for a in range(len(self.theta)):
            l = []
            for i,y in enumerate(y_centers):
                s = np.random.multinomial(1,self.probabilities[a][y])
                l_ai = self.space[np.where(s==1)]
                l.append(l_ai)
            L.append(l)
        return L 
