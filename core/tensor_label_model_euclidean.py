import numpy as np
from core.tensor_decomp import mixture_tensor_decomp_full, mse_perm, mse


class TensorLabelModelEuclidean:
    def __init__(self,k,Y_emb_unique) -> None:
        
        self.k  = k
        self.thetas = None
        self.Y_emb_unique = Y_emb_unique
        self.dist_fun = lambda x,y: np.linalg.norm(x-y)**2

    def mu_recovery(self, L_emb, Y_emb_unique, triplet_idx_a, triplet_idx_b, triplet_idx_c):
        """Recover mu for a single labeling function (index triplet_index_a)
        Follows the multi-view models approach (Section 3.3)
        Constructs symmetric matrices / tensors from observed quantities

        Parameters
        ----------
        triplet_idx_a, triplet_idx_b, triplet_idx_c
            Indices for the three labeling functions

        """
        # setups for base matries and tensors
            
        ###### Embed the entire label space ######

        debug = False 

        m,n,d = L_emb.shape 
        max_m = m 
        k = self.k 

        (w_rec, mu_hat_1, mu_hat_2, mu_hat_3,) = mixture_tensor_decomp_full(
            w= np.ones(n) / n,
            x1=L_emb[0, :, :].T,
            x2=L_emb[1, :, :].T,
            x3=L_emb[2, :, :].T,
            k=k,
            debug=debug,
            # savedir="T_pos_hat",
        )
        
        mu_hat = np.array([mu_hat_1, mu_hat_2, mu_hat_3])
        
        ### Using TD ###
        print(mu_hat.shape,Y_emb_unique.shape)
        
        exp_sq_dist_TD = np.zeros((max_m, k))

        for lf in range(max_m):
            exp_sq_dist_TD[lf] = np.abs(
                (np.linalg.norm(L_emb[lf, :, :], axis=1) ** 2).mean()
                + (np.linalg.norm(Y_emb_unique, axis=1) ** 2)
                - 2 * (mu_hat[lf, :, :] * Y_emb_unique.T).sum(axis=0)
            )
        #w_rec = np.ones(len(w_rec))*(1/k)
        #print('w_rec',w_rec)
        exp_sq_dist_TD = w_rec @ exp_sq_dist_TD.T
        exp_sq_dist_TD = exp_sq_dist_TD/np.sum(exp_sq_dist_TD)
        #print('expected distance',exp_sq_dist_TD)
        thetas_td = np.ones_like(exp_sq_dist_TD) / exp_sq_dist_TD
        
        #print('thetas',thetas_td)

        self.thetas = thetas_td
        return thetas_td

    def predict(self,L_emb,abstain_allowed=False):
        
        # Compute the weighted Fréchet mean
        m,n,d = L_emb.shape 
        thetas= self.thetas
        Y_emb_unique = self.Y_emb_unique
        k = self.k 
        preds = []

        thetas = thetas/np.sum(thetas)
        
        for i in range(n):
            mean_l = np.sum([thetas[a]*L_emb[a][i] for a in range(m)],axis=0)
            
            if(not abstain_allowed):
                dists  = [self.dist_fun(mean_l,yj) for yj in Y_emb_unique]
            else:
                dists  = [self.dist_fun(mean_l,yj) for yj in Y_emb_unique[:-1]]
            
            preds.append(np.argmin(dists))

        return preds
    
    