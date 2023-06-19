import sys 
sys.path.append('../')
sys.path.append('../../')

import scipy
import os 
import json
import networkx as nx 
from scipy import stats

from core.exp_label_model import * 
from core.pse import pseudo_embedding 

from core.tensor_label_model_euclidean import TensorLabelModelEuclidean
from core.tensor_label_model_pse import TensorLabelModelPSE

from multiprocessing import Process
from core.tensor_label_model_pse import TensorLabelModelPSE

from core.exp_label_model import * 
from core.pse import pseudo_embedding 

from collections import defaultdict 

def dist_fun_0_1(y_1,y_2):
    return int(y_1 != y_2)

def draw_samples(n_samples,theta,D):
    k = len(D)
    space= np.arange(0,k,1)

    exp_model = ExponentialLabelModel()
    P = exp_model.get_probability_table_true(space, theta, D)

    Y = np.array([np.random.randint(k) for i in range(n_samples)])

    L = exp_model.draw_samples(Y)
    L = np.array(L)
    L = L.squeeze()

    acc_mv = run_mv(L,Y)
    acc_pse =  run_tensor_lm_pse(L,Y,D,k,dim=3)
    acc_1_hot = run_tensor_lm_1_hot(L,Y,k)
    return acc_mv,acc_pse,acc_1_hot


def run_flying_squid_binary(L,Y,abstain_allowed=False):
    from flyingsquid.label_model import LabelModel
    L_ = np.copy(L)
    L_ = L_.T 
    if(abstain_allowed):
        L_[np.where(L_==0)]=-2  # original 0 is negative class
        L_[np.where(L_==2)]=0  # original 2 is abstain
        L_[np.where(L_==-2)]=-1  

    else:
        L_[np.where(L_==0)]=-1
        #L_[np.where(L_==-1)]=0

    m,n = L.shape
    label_model = LabelModel(m)

    label_model.fit(L_)
    Y_ = np.copy(Y)

    #Y_[Y_==0]=-1
    #print(L_)
    preds = label_model.predict(L_).reshape(Y_.shape)

    preds[preds==-1] = 0
    accuracy = np.sum(preds == Y_) / Y_.shape[0]

    print('Label model accuracy of flying squid: {}%'.format(int(100 * accuracy)))
    return accuracy,preds

def run_flying_squid_multi_class(L,Y,k,abstain_allowed=False):
    from flyingsquid.label_model import LabelModel
    
    m,n = L.shape

    label_model = LabelModel(m)
    probs = np.zeros((n,k))

    for j in range(k):
        L_ = np.copy(L)
        L_ = L_.T
        L2 = L.T
        if(abstain_allowed):
            #L_[np.where((L_!=j)&(L_!=k))] = -2 # not abstained but assigned some other class
            L_[np.where(L2==j)] = 1
            L_[np.where(L2==k)] = 0        # abstained
            L_[np.where(L2!=j)] = -1 #  not abstained but assigned some other class

        else:
            L_[np.where(L2!=j)] = -1
            L_[np.where(L2==j)] = 1
        
        label_model.fit(L_)

        combination_probs = label_model.predict_proba(L_, verbose=False)
        preds = label_model.predict(L_).reshape(Y.shape)
        print(combination_probs.shape)
        probs[:,j] = combination_probs[:,1]

    preds_final = np.argmax(probs,axis=1)

    accuracy = np.sum(preds_final == Y) / Y.shape[0]

    print('Label model accuracy of flying squid: {}%'.format(int(100 * accuracy)))
    return accuracy, preds_final

def run_flying_squid(L,Y,k,abstain_allowed=False):
    if(k>2):
        acc_fs,preds = run_flying_squid_multi_class(L,Y,k,abstain_allowed)
    else:
        acc_fs,preds = run_flying_squid_binary(L,Y,abstain_allowed)
    return acc_fs,preds 


def run_tensor_lm_1_hot(L,Y,k,abstain_allowed=False):
    
    Y_emb_unique = np.eye(k)[[np.arange(k)]].squeeze()

    print(Y_emb_unique.shape)
    L_emb =[[Y_emb_unique[l,:] for l in L_i] for L_i in L]
    L_emb = np.array(L_emb)
    L_emb.shape 
    tlm = TensorLabelModelEuclidean(k,Y_emb_unique)
    tlm.k = k
    
    tlm.mu_recovery(L_emb,Y_emb_unique,0,1,2)
    
    Y_hat = tlm.predict(L_emb,abstain_allowed)
    accuracy = np.sum(Y_hat == Y) / Y.shape[0]
    print('Label model accuracy of tensor lm 1-hot: {}%'.format(int(100 * accuracy)))
    return accuracy,Y_hat

#run_tensor_lm_1_hot(L,y_centers,k)
def accuracy_dist(y,y_hat,D):
    y = np.array(y).astype(int)
    y_hat = np.array(y_hat).astype(int)

    n = len(y)
    s = [D[y_hat[i]][y[i]] for i in range(n)]
    return sum(s)/n 


def run_mv(L,Y):
    mv_out  = scipy.stats.mode(L, axis=0)[0].squeeze()
    accuracy = np.sum(mv_out == Y) / Y.shape[0]
    print('Majority vote accuracy: {}%'.format(int(100 * accuracy)))
    return accuracy,mv_out 

def run_tensor_lm_pse(L,Y,D,k,dim=3):

    Y_emb_unique, tk = pseudo_embedding(D, dim)

    L_emb =[[Y_emb_unique[l,:] for l in L_i] for L_i in L]
    L_emb = np.array(L_emb)
    L_emb.shape

    tlm = TensorLabelModelPSE(k,Y_emb_unique,tk)
    tlm.k = k
    tlm.mu_recovery(L_emb,0,1,2)

    Y_hat = tlm.predict(L_emb)

    accuracy = np.sum(Y_hat == Y) / Y.shape[0]

    print('Label model accuracy of tensor lm pse: {}%'.format(int(100 * accuracy)))
    return accuracy ,Y_hat



def get_star_tree(n_branches,branch_size):
  n = n_branches*branch_size - (n_branches-1)
  N=1
  A = np.zeros((n,n))
  for i in range(n_branches):
    
    branch_nodes = [0] + [j for j in range(N,N+branch_size-1)]
    #print(branch_nodes)
    b = branch_nodes
    for j in range(branch_size-1):
      #print(j)
      A[b[j]][b[j+1]] = 1
      A[b[j+1]][b[j]] = 1
    N = N + branch_size-1
  return A


def run_tree(conf):

    thetas = conf['thetas']
    n_samples = conf['n_samples']
    branch_size = conf['branch_size']

    np.random.seed(conf['seed'])
    
    A = get_star_tree(3,branch_size)
    G = nx.from_numpy_array(A)
    D_tree = nx.floyd_warshall_numpy(G)
    X, tk = pseudo_embedding(D_tree*D_tree, dim=3)
    D = D_tree*D_tree

    k = len(D)
    space= np.arange(0,k,1)

    exp_model = ExponentialLabelModel()
    P = exp_model.get_probability_table_true(space, thetas, D)

    Y = np.array([np.random.randint(k) for i in range(n_samples)])

    L = exp_model.draw_samples(Y)
    L = np.array(L)
    L = L.squeeze()

    acc_mv, Y_hat_mv = run_mv(L,Y)
    acc_pse,Y_hat_pse =  run_tensor_lm_pse(L,Y,D,k,dim=3)
    acc_1_hot,Y_hat_1_hot = run_tensor_lm_1_hot(L,Y,k)
    acc_fs,Y_hat_fs = run_flying_squid(L,Y,k,abstain_allowed=False)

    acc_mv_dist = accuracy_dist(Y,Y_hat_mv,D)
    acc_pse_dist = accuracy_dist(Y,Y_hat_pse,D)
    acc_1_hot_dist = accuracy_dist(Y,Y_hat_1_hot,D)
    acc_fs_dist = accuracy_dist(Y,Y_hat_fs,D)

    out = {'acc_mv':acc_mv,'acc_pse':acc_pse,'acc_1_hot':acc_1_hot,'acc_fs':acc_fs}
    out['acc_mv_dist'] = acc_mv_dist
    out['acc_pse_dist'] =acc_pse_dist
    out['acc_1_hot_dist'] =acc_1_hot_dist
    out['acc_fs_dist'] = acc_fs_dist

    out.update({'k':k,'n_samples':n_samples})
    
    with open(conf['out_file_path'], 'w') as fout:
        json_dumps_str = json.dumps(out, indent=4)
        print(json_dumps_str, file=fout)

    return acc_mv, acc_pse,acc_1_hot


def run_complete_graph_0_1(conf):

    thetas = conf['thetas']
    n_samples = conf['n_samples']
    #branch_size = conf['branch_size']
    k = int(conf['k'])

    np.random.seed(conf['seed'])
    D = np.ones((k,k)) - np.eye(k)

    space= np.arange(0,k,1)

    exp_model = ExponentialLabelModel()
    P = exp_model.get_probability_table_true(space, thetas, D)

    Y = np.array([np.random.randint(k) for i in range(n_samples)])

    L = exp_model.draw_samples(Y)
    L = np.array(L)
    L = L.squeeze()

    acc_mv, Y_hat_mv = run_mv(L,Y)
    #acc_pse,Y_hat_pse =  run_tensor_lm_pse(L,Y,D,k,dim=3)
    acc_1_hot,Y_hat_1_hot = run_tensor_lm_1_hot(L,Y,k)
    acc_fs,Y_hat_fs = run_flying_squid(L,Y,k,abstain_allowed=False)

    acc_mv_dist = accuracy_dist(Y,Y_hat_mv,D)
    #acc_pse_dist = accuracy_dist(Y,Y_hat_pse,D)
    acc_1_hot_dist = accuracy_dist(Y,Y_hat_1_hot,D)
    acc_fs_dist = accuracy_dist(Y,Y_hat_fs,D)

    out = {'acc_mv':acc_mv,'acc_1_hot':acc_1_hot,'acc_fs':acc_fs}
    out['acc_mv_dist'] = acc_mv_dist
    #out['acc_pse_dist'] =acc_pse_dist
    out['acc_1_hot_dist'] =acc_1_hot_dist
    out['acc_fs_dist'] = acc_fs_dist

    out.update({'k':k,'n_samples':n_samples})
     
    with open(conf['out_file_path'], 'w') as fout:
        json_dumps_str = json.dumps(out, indent=4)
        print(json_dumps_str, file=fout)

    return None


def par_run(run_conf,lst_confs,overwrite=True):
    lstP = []
    print(len(lst_confs))
    for conf in lst_confs:
        #print(conf)
        #conf = copy.deepcopy(conf) # ensure no shit happens
        p = Process(target = run_conf, args=(conf,))
        p.start()
        
        lstP.append(p)
    for p in lstP:
        p.join()

def assign_devices_to_confs(lst_confs,lst_devices = ['cpu']): 
    #round robin
    i = 0
    n = len(lst_confs)
    while(i<n):
        for dev in lst_devices:
            if(i<n):
                lst_confs[i]['device'] = dev
            i+=1
    
def exclude_existing_confs(lst_confs):
    lst_out_confs = []
    for conf in lst_confs:
        path = conf["out_file_path"]
        if os.path.exists(path):
            print(f"path exists {conf['out_file_path']}")
        else:
            lst_out_confs.append(conf)
    return lst_out_confs


def batched_par_run(run_conf,lst_confs,batch_size=2, lst_devices=['cpu'],overwrite=True):
    
    if(not overwrite):
        lst_confs = exclude_existing_confs(lst_confs)
        n = len(lst_confs)
        print(f'NUM confs to run : {n}')

    assign_devices_to_confs(lst_confs,lst_devices)

    i=0
    n = len(lst_confs)
    while(i<n):
        print(f'running confs from {i} to {i+batch_size} ')
        #for conf in lst_confs[i:i+batch_size]:
        #    print(conf['device'])
        par_run(run_conf,lst_confs[i:i+batch_size],overwrite)
        i+=batch_size 
    

def seq_run(lst_confs,run_conf):
    for conf in lst_confs:
        run_conf(conf)

