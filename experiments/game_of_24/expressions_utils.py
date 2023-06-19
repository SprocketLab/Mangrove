import itertools
import numpy as np 

token_dict = {'(':0,')':1,'+':2,'-':3,'x':4,'/':5}
ops = ['+','-','x','/']
t1 = '( ( n1 o1 n2 ) o2 ( n3 o3 n4 ) )'
t2 = '( ( ( n1 o1 n2 ) o2 n3 ) o3 n4 )'
t3 = '( n1 o1 ( n2 o2 ( n3 o3 n4 ) ) )'

def generate_all_trees():
    def replace(t,o1,o2,o3):
        return t.replace('o1',o1).replace('o2',o2).replace('o3',o3)
    
    lst_T = []
    for o1 in ops:
        for o2 in ops:
            for o3 in ops:
                lst_T.extend([replace(t1,o1,o2,o3),replace(t2,o1,o2,o3),replace(t3,o1,o2,o3)])
    return lst_T 

def eval_exp(t,n1,n2,n3,n4):
    t = t.split(' ')
    def eval_op(x,y,o):
        if(o=='+'):
            return x+y 
        elif(o=='-'):
            return x-y 
        elif(o=='x'):
            return x*y 
        elif(o=='/'):
            if(y==0):
                return 1000
            else:
                return x/y 

    d = {'n1':n1,'n2':n2,'n3':n3,'n4':n4}
    s = []
    v = 0
    for c in t:
        if(c=='('):
            s.append(c)
        elif(c in ['n1','n2','n3','n4']):
            s.append(d[c])
        elif(c in ['+','-','x','/']):
            s.append(c)

        elif(c==')'):
            y = s.pop()
            o = s.pop()
            x = s.pop()
            _ = s.pop()
            z = eval_op(x,y,o)
            s.append(z)
    return s[0]

def eval_exp_permute(ex, n1,n2,n3,n4):
    lst_p = list(itertools.permutations([n1,n2,n3,n4]))
    return [ eval_exp(ex,p[0],p[1],p[2],p[3]) for p in lst_p], lst_p 

def compute_distances(lst_T,n1,n2,n3,n4):
    n = len(lst_T)
    D = np.zeros((n,n))
    z = np.zeros(n)

    for i in  range(len(lst_T)):
        lst_v_p, lst_p = eval_exp_permute(lst_T[i],n1,n2,n3,n4)
        z[i] = np.mean(lst_v_p)
        
        #z = [eval_exp(t,n1,n2,n3,n4) for t in lst_T]
    
    for i in range(n):
        for j in range(i,n):

            #D[i][j] = min(10,abs(z[i]-z[j])*10)
            D[i][j] = abs(z[i]-z[j])#/(max(1e-3,abs(z[i]+z[j])))
            D[j][i] = D[i][j]

    return D 
