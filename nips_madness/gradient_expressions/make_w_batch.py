import theano
import theano.tensor as T
import numpy as np

sign = theano.shared(np.array([[1,-1],[1,-1]]),name = "sign")

def make_W_with_x(Z,J,D,S,N,X):

    '''

    creates a symbolic expression for W given the parameters, the specific values of Z, and an array X which provides the numberical values for distance between sites

    '''
    
    #we want the tensor to be [nz, 2N,2N] in the end
    #lets start by making an [nz,2,2,N,N] matrix

    j = T.reshape(sign*J,[1,2,1,2,1])
    d = T.reshape(sign*D,[1,2,1,2,1])
    s = T.reshape(S,[1,2,1,2,1])

    z = T.reshape(Z,[-1,2,N,2,N])

    xt = T.reshape(X,[1,-1])

    xx = T.reshape((xt - xt.T),[1,1,N,1,N])

    wnn = T.exp(-(xx)**2/(2 * s**2))
    #[2,N,2,N]

    W = wnn*(j + d*z)

    return T.reshape(W,[-1,2*N,2*N])

def make_WJ_with_x(Z,J,D,S,N,X,dj):

    '''

    creates a symbolic expression for W given the parameters, the specific values of Z, and an array X which provides the numberical values for distance between sites

    '''
    
    j = T.reshape(sign*J,[1,2,1,2,1])
    d = T.reshape(sign*D,[1,2,1,2,1])
    s = T.reshape(S,[1,2,1,2,1])

    z = T.reshape(Z,[-1,2,N,2,N])

    xt = T.reshape(X,[1,-1])

    xx = T.reshape((xt - xt.T),[1,1,N,1,N])
    
    wnn = T.exp(-(xx)**2/(2 * s**2))
    #[2,N,2,N]

    jt = T.reshape(dj,[1,2,1,2,1,2,2])

    W = T.reshape(wnn,[1,2,N,2,N,1,1])*jt

    return T.reshape(W,[1,2*N,2*N,2,2])#this one is independent of z!

def make_WD_with_x(Z,J,D,S,N,X,dd):

    '''

    creates a symbolic expression for W given the parameters, the specific values of Z, and an array X which provides the numberical values for distance between sites

    '''
    
    #we want the tensor to be [nz, 2N,2N] in the end
    #lets start by making an [nz,2,2,N,N] matrix

    j = T.reshape(sign*J,[1,2,1,2,1])
    d = T.reshape(sign*D,[1,2,1,2,1])
    s = T.reshape(S,[1,2,1,2,1])

    z = T.reshape(Z,[-1,2,N,2,N])

    xt = T.reshape(X,[1,-1])

    xx = T.reshape((xt - xt.T),[1,1,N,1,N])

    wnn = T.exp(-(xx)**2/(2 * s**2))
    #[2,N,2,N]

    dt = T.reshape(dd,[-1,2,1,2,1,2,2])

    W = T.reshape(wnn*z,[-1,2,N,2,N,1,1])*dt

    return T.reshape(W,[-1,2*N,2*N,2,2])

def make_WS_with_x(Z,J,D,S,N,X,ds):

    '''

    creates a symbolic expression for W given the parameters, the specific values of Z, and an array X which provides the numberical values for distance between sites

    '''
    
    #we want the tensor to be [nz, 2N,2N] in the end
    #lets start by making an [nz,2,2,N,N] matrix

    j = T.reshape(sign*J,[1,2,1,2,1])
    d = T.reshape(sign*D,[1,2,1,2,1])
    s = T.reshape(S,[1,2,1,2,1])

    z = T.reshape(Z,[-1,2,N,2,N])

    xt = T.reshape(X,[1,-1])

    xx = T.reshape((xt - xt.T),[1,1,N,1,N])

    wnn = T.exp(-(xx)**2/(2 * s**2))
    #[2,N,2,N]

    st = T.reshape(ds,[1,2,1,2,1,2,2])

    W = T.reshape(wnn*(2*(xx)**2/(2 * s**3))*(j + d*z),[2,N,2,N,1,1])*st

    return T.reshape(W,[-1,2*N,2*N,2,2])
    

def test_x(x):
    xx = T.reshape(x,[-1,1])

    return xx - xx.T

if __name__ == "__main__":

    J = theano.shared(np.array([[1.,1.],[1.,1.]]),name = "j")
    D = theano.shared(np.array([[1.,1.],[1.,1.]]),name = "d")
    S = theano.shared(np.array([[1.,1.],[1.,1.]]),name = "s")

    n = theano.shared(50,name = "n_sites")
    nz = theano.shared(10,name = 'n_samples')

    X = theano.shared(np.linspace(-1,1,n.get_value()),name = "positions")

    Z = T.tensor3("z","float32")

    ww = make_W_with_x(Z,J,D,S,n,X)

    dwdj = T.reshape(T.jacobian(T.flatten(ww),J),[nz,2*n,2*n,2,2])
    dwdd = T.reshape(T.jacobian(T.flatten(ww),D),[nz,2*n,2*n,2,2])
    dwds = T.reshape(T.jacobian(T.flatten(ww),S),[nz,2*n,2*n,2,2])

    DWJ = theano.function([Z],dwdj,allow_input_downcast = True)
    DWD = theano.function([Z],dwdj,allow_input_downcast = True)
    DWS = theano.function([Z],dwdj,allow_input_downcast = True)

    ##make our Z vector
    NN = n.get_value()
    NZ = nz.get_value()

    zz = np.random.rand(NZ,NN,NN)
    ###
    
    WJ = DWJ(zz)
    WD = DWD(zz)
    WS = DWS(zz)

    print(WJ.shape)
    print(WD.shape)
    print(WS.shape)
