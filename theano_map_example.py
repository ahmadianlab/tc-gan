import lasagne as L
import numpy as np
import theano
import theano.tensor as T


slope = theano.shared(0.0)
X = T.matrix('x', 'float32')
Y = T.matrix('y', 'float32')


def loss(x, y):
    return (slope * x - y)**2

total_loss = theano.map(loss, sequences=[X, Y])[0].sum()
UP = L.updates.adam(total_loss, [slope])
train = theano.function([X, Y], total_loss,
                        updates=UP, allow_input_downcast=True)


x_data = np.linspace(-1, 1, 100).reshape((5, -1))
y_data = x_data * 0.123

for _ in range(10):
    for _ in range(100):
        num_loss = train(x_data, y_data)
    print("slope", slope.get_value(), "loss", num_loss)
