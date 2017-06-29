import numpy as np
import time

def KS_distance(A,B):
    
    L = [[a,0] for a in A] + [[b,1] for b in B]

    L.sort(key = lambda x:x[0])

    La = len(A)
    Lb = len(B)

    n = [0.,0.]

    dist = 0

    for k in L:
        n[k[1]] += 1.

        d = np.abs((n[0]/La) - (n[1]/Lb))

        if d > dist:
            dist = d

    return dist

def complexity(inp):

    A = (inp - inp.mean())/(inp.max() - inp.min() + .001)
    
    b1 = np.abs(A[:-1] - A[2:])
    b2 = np.abs(A[:,:-1] - A[:,2:])
    b2 = np.abs(A[:,:,:-1] - A[:,:,2:])

    B = np.flatten(np.concatenate([b1,b2,b3]),[-1])

    return np.std(B)



def coding_level(A,source):
    assert source in ("data","model")

    if source == "data":
        return np.float32(len([a for a in A if a > 5]))/len(A)
    elif source == "model":
        return np.float32(len([a for a in A if a > 0]))/len(A)

def comscore(A):
    

def main(n):
    print(KS_distance(np.random.uniform(-1,1,n),np.random.uniform(-1,1,n)))

if __name__ == "__main__":
    for k in [10,100,1000,10000]:
        t1 = time.time()
        main(k)
        print("time",time.time() - t1)
