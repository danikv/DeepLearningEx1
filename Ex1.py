import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def softmax_gradient(X, Y_true, Y_predicted):
    M = X.shape[1]
    grad = Y_predicted - Y_true
    return 1/M * X.dot((Y_predicted - Y_true).T)

def accuracy(Y_true, Y_predicted):
    return np.sum(np.argmax(Y_true, axis=0) == np.argmax(Y_predicted, axis=0)) / Y_true.shape[1]

def SGD(model, X, Y, epoch):
    loss = []
    accuracy_sgd = []
    for i in range(epoch):
        Y_predicted = model(X)
        gradient = softmax_gradient(X, Y, Y_predicted)
        model.update_weigths(gradient)
        accuracy_sgd.append(accuracy(Y, Y_predicted))
        loss.append(cross_entropy_loss(Y, Y_predicted))
    return loss, accuracy_sgd

class softmax_layer:
    def dim_in(self):
        return 0
    
    def dim_out(self):
        return 0
    
    def update_weigths(self, gradient, learning_rate):
        return gradient
    
    def __call__(self, x):
        #exp = np.exp(x.T - np.max(x, axis=1))
        exp = np.exp(x)
        return (exp.T / np.sum(exp, axis=1))

class linear_layer:
    def __init__(self, dim_in, dim_out):
        self._A = np.random.rand(dim_in,dim_out)
        self._B = np.random.rand(dim_out)
        self._dim_in = dim_in
        self._dim_out = dim_out
        
    def update_weigths(self, gradient, learning_rate):
        self._A = self._A - learning_rate * gradient
        return gradient
    
    def dim_in(self):
        return self._dim_in
    
    def dim_out(self):
        return self._dim_out
    
    def __call__(self, x):
        return x.T.dot(self._A) + self._B

def cross_entropy_loss(y_true, y_predicted, epsilon=1e-10):
    predictions = np.clip(y_predicted, epsilon, 1. - epsilon)
    M = predictions.shape[1]
    return -np.sum(y_true * np.log(predictions)) / M

class sequential_model:
    def __init__(self, *layers, learning_rate=0.1):
        self._learning_rate = learning_rate
        self._layers = []
        last_dim_out = 0
        for layer in layers:
            if last_dim_out != 0 and layer.dim_in() != 0 and last_dim_out != layer.dim_in():
                print('dimension dont match layer out dim {} , next layer dim in {}'.format(last_dim_out, layer.dim_in()))
                raise 
            self._layers.append(layer)
            if layer.dim_out() != 0:
                last_dim_out = layer.dim_out()
                
    def update_weigths(self, gradient):
        for layer in reversed(self._layers):
            gradient = layer.update_weigths(gradient, self._learning_rate)
            
    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

model = sequential_model(
        linear_layer(2, 2),
        softmax_layer()
    )

mat = scipy.io.loadmat('SwissRollData.mat')
X = mat['Yt']
Y = mat['Ct']
print(X.shape)

Y_predicted = model(X)
print(Y.shape)
print(Y_predicted.shape)
print(Y_predicted.T)

print(cross_entropy_loss(Y, Y_predicted))

loss,accuracy = SGD(model, X, Y, 1000)

#fig = plt.figure(figsize=(10,6))

fig, axs = plt.subplots(2)
axs[0].plot(loss)
#ax.plot(accuracy)
axs[0].set_ylabel('loss')

axs[1].plot(accuracy)
#ax.plot(accuracy)
axs[1].set_ylabel('accuracy')

plt.show()