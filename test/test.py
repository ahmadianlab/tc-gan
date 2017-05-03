import theano
import theano.tensor as T
import lasagne as L
import numpy as np

O = theano.shared(3.)
x = T.scalar("x",'float32')

func = (x - O)*(x - O)
gexp = -2*(x - O)

F = theano.function([x],func,allow_input_downcast = True)
G = theano.function([x],gexp,allow_input_downcast = True)

print("Oval {}".format(O.get_value()))
print("F test {}".format(F(1)))
print("G test {}".format(G(1)))

UP = L.updates.adam([gexp],[O],1.)

print("Made Updates")

train = theano.function([x],func,updates = UP,allow_input_downcast = True)

X = np.random.rand()
print("X is : {}".format(X))

for k in range(100):
    train(X)
    if k % 10 == 0:
        print(O.get_value())
        
