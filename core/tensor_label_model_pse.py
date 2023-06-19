
import numpy as np
import itertools

from core.tensor_decomp import mixture_tensor_decomp_full, mse_perm, mse


class TensorLabelModelPSE:
    def __init__(self,k,Y_emb_unique,tk) -> None:
        
        self.k  = k 
        self.Y_emb_unique = Y_emb_unique
        self.dist_fun = lambda x,y,tk : np.linalg.norm(x[:tk]-y[:tk])**2 - np.linalg.norm(x[tk:]-y[tk:])**2
        self.tk = tk 

    def mu_recovery(self, L_emb, triplet_idx_a, triplet_idx_b, triplet_idx_c):
        """Recover mu for a single labeling function (index triplet_index_a)
        Follows the multi-view models approach (Section 3.3)
        Constructs symmetric matrices / tensors from observed quantities

        Parameters
        ----------
        triplet_idx_a, triplet_idx_b, triplet_idx_c
            Indices for the three labeling functions

        """
        ###### Embed the entire label space ######

        debug = False 

        m,n,d = L_emb.shape 
        max_m = m 
        

        k = self.k 
        tk = self.tk 

        Y_emb = self.Y_emb_unique
        Yspace_emb = Y_emb

        Y_emb_pos, Y_emb_neg = Y_emb[:, :tk], Y_emb[:, tk:]
        print('pos shape', Y_emb_pos.shape)
        print('neg shape',Y_emb_neg.shape)

        noneuclidean = np.prod(Y_emb_neg.shape) > 0

        ###### Sample proportionally to the probabilities ######
        #L = sample_LFs(Yspace, Y_inds, probs, max_m)

        ###### Get sampled LF embeddings ######
        
        L_emb_pos, L_emb_neg = L_emb[:, :, :tk], L_emb[:, :, tk:]

        ###### Tensor decomposition for + - ######
        #print(L_emb_pos.shape)
        (w_rec, mu_hat_pos1, mu_hat_pos2, mu_hat_pos3,) = mixture_tensor_decomp_full(
            np.ones(n) / n,
            L_emb_pos[0, :, :].T,
            L_emb_pos[1, :, :].T,
            L_emb_pos[2, :, :].T,
            k=k,
            debug=debug,
            # savedir="T_pos_hat",
        )
        w_rec_pos = w_rec
        mu_hat_pos = np.array([mu_hat_pos1, mu_hat_pos2, mu_hat_pos3])

        if noneuclidean:
            print(L_emb_neg.shape)
            (w_rec, mu_hat_neg1, mu_hat_neg2, mu_hat_neg3,) = mixture_tensor_decomp_full(
            np.ones(n) / n,
            L_emb_neg[0, :, :].T,
            L_emb_neg[1, :, :].T,
            L_emb_neg[2, :, :].T,
            k=k,
            debug=debug)

          
            w_rec_neg = w_rec
            mu_hat_neg = np.array([mu_hat_neg1, mu_hat_neg2, mu_hat_neg3])
        
        Y_emb_unique = Y_emb
        Y_emb_unique_pos, Y_emb_unique_neg = Y_emb_unique[:, :tk], Y_emb_unique[:, tk:]
        #Yspace_emb_pos, Yspace_emb_neg = Yspace_emb[:, :tk], Yspace_emb[:, tk:]
        #print('w_rec_pos',w_rec_pos)
        #print('w_rec_neg',w_rec_neg)

        ### Using TD ###
        exp_sq_dist_TD_pos = np.zeros((max_m, k))
        for lf in range(max_m):
            exp_sq_dist_TD_pos[lf] = (
                (np.linalg.norm(L_emb_pos[lf, :, :], axis=1) ** 2).mean()
                + np.linalg.norm(Y_emb_unique_pos, axis=1) ** 2
                - 2 * (mu_hat_pos[lf, :, :] * Y_emb_unique_pos.T).sum(axis=0)
            )
        #print("using TD (+)")
        #print(w_rec_pos @ exp_sq_dist_TD_pos.T)

        if noneuclidean:
            exp_sq_dist_TD_neg = np.zeros((max_m, k))
            for lf in range(max_m):
                exp_sq_dist_TD_neg[lf] = (
                    (np.linalg.norm(L_emb_neg[lf, :, :], axis=1) ** 2).mean()
                    + np.linalg.norm(Y_emb_unique_neg, axis=1) ** 2
                    - 2 * (mu_hat_neg[lf, :, :] * Y_emb_unique_neg.T).sum(axis=0)
                )
            #print("using TD (-)")
            #print(w_rec_neg @ exp_sq_dist_TD_neg.T)

        if noneuclidean:
            #print("using TD (pos-neg)")
            exp_sq_dist_TD = (w_rec_pos @ exp_sq_dist_TD_pos.T) - (
                w_rec_neg @ exp_sq_dist_TD_neg.T
            )
            #print(exp_sq_dist_TD)
        else:
            #print("using TD (pos)")
            exp_sq_dist_TD = w_rec_pos @ exp_sq_dist_TD_pos.T
            #print(exp_sq_dist_TD)

        
        exp_sq_dist_TD = exp_sq_dist_TD/np.sum(exp_sq_dist_TD)

        thetas_td = np.ones_like(exp_sq_dist_TD) / exp_sq_dist_TD
        
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
            #print(mean_l.shape)
            if(not abstain_allowed):
                dists  = [self.dist_fun(mean_l,yj,self.tk) for yj in Y_emb_unique]
            else:
                dists  = [self.dist_fun(mean_l,yj,self.tk) for yj in Y_emb_unique[:-1]]
            
            preds.append(np.argmin(dists))
        
        return preds
    
    