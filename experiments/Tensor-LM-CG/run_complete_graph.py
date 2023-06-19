
import sys 
sys.path.append('../')
from run_lib import * 

if __name__ == "__main__":
    lst_n_samples = [10000]
    lst_k = np.arange(2,11,1)
    thetas = np.array([4,0.5,0.5])
    
    lst_confs = []
    T = 100
    out_dir = '../../outputs/cg_runs/5/'

    ex = os.path.exists(out_dir)
    if not ex:
        os.makedirs(out_dir)

    pfx  = f'{out_dir}/cg_runs'
    lst_confs = []
    for n_samples in lst_n_samples: 
        for k in lst_k : 
            for t in range(T):
                conf = {'n_samples':n_samples,'thetas':thetas, 'k':k,'out_file_path':f'{pfx}_{n_samples}_{k}_{t}.json'}
                conf['seed'] = t 
                conf['trial_id'] = t 
                lst_confs.append(conf)
    
    print(f'total confs : {len(lst_confs)}')

    batched_par_run(run_complete_graph_0_1,lst_confs,batch_size=20, lst_devices=['cpu'],overwrite=False)

