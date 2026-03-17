
from core.Value import Value
import numpy as np

class Neuron:
    def __init__(self, nin):
        #nin is the number of inputs in the neuron
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w,x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    #nin - no of inputs
    #nout - list number - defines sizes of all wanted layers
    def __init__(self,nin, nout):
        sizes = [nin] + nout
        self.layers = [Layer(sizes[i], sizes[i+1])for i in range (len(sizes) - 1)]
    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x) # we have an object of class layer here tat is being used to input x
        return x

    def parameters(self):
        return [ p for layer in self.layers for p in layer.parameters()]
        