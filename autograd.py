# Implementation of a basic AutoGrad.

import math

class Value:

    def __init__(self, data, parent_nodes = ()):
        self.data = data
        self.gradient = 0.0
        # A variable that store an empty function. This prevents _backpropagation() call for the leaf nodes.
        self._back_propagation = lambda: None                                                
        self._parent_node = set(parent_nodes)

    # IMPLEMENTING ADDITION.
    def __add__(self, other):
        # If other is anything else than an object of class Value (like '2' in expression 'a+2'),
        # we make it one.
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other))

        def _back_propagation():
            # '+=' for Gradient Accumulation.
            self.gradient += output.gradient    
            other.gradient += output.gradient

        output._back_propagation = _back_propagation

        return output

    # A routing function for a condition like '2+a'.
    def __radd__(self, other):
        return self + other

    # IMPLEMENTING MULTIPLICATION.
    def __mul__(self, other):
        # If other is anything else than an object of class Value (like '2' in expression 'a*2'),
        # we make it one.
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other))

        def _back_propagation():
            # '+=' for Gradient Accumulation.
            self.gradient += other.data * output.gradient
            other.gradient += self.data * output.gradient

        output._back_propagation = _back_propagation

        return output

    # A routing function for a condition like '2*a'.
    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        tanh_result = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        output = Value(tanh_result, (self,))

        def _back_propagation():
            # I am skeptical if gradient accumulation is needed here.
            self.gradient += (1 - tanh_result**2) * output.gradient     

        output._back_propagation = _back_propagation
        
        return output
        
    # Function to automatically start Back-propagation from the output node.
    def back_propagation(self):

        # Implementing Topological Sort
        sorted_nodes = []
        visited_nodes = set()

        def topological_sort(node):

            if node not in visited_nodes:
                visited_nodes.add(node)

                for parent in node._parent_nodes:
                    topological_sort(parent)

                sorted_nodes.append(node)

        topological_sort(self)

        # Defining back_propagation()

        # Setting the gradient of Output Node as 1.0
        self.gradient = 1.0    

        for node in reversed(sorted_nodes):
            node._back_propagation()