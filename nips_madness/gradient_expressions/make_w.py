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

    j = T.reshape(sign*J,[2,1,2,1])
    d = T.reshape(sign*D,[2,1,2,1])
    s = T.reshape(S,[2,1,2,1])

    z = T.reshape(Z,[2,N,2,N])

    xt = T.reshape(X,[1,-1])

    xx = T.reshape((xt - xt.T),[1,N,1,N])

    wnn = T.exp(-(xx)**2/(2 * s**2))
    #[2,N,2,N]

    W = wnn*(j + d*z)

    return T.reshape(W,[2*N,2*N])
    

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
