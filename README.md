JSNeuralNet
===========

A object oriented JavaScript implementation of a neural network.

The neural network uses the sigmoid function as a squashing function for neuron output. Neurons are biased and also implement change momentum. The network uses the Backpropagation algorithm described here to implement training:

http://en.wikipedia.org/wiki/Backpropagation

How to Use
==========
Using the neural network is straight forward. Here is an example that adds a third, hidden layer with two neurons, and attempts to approximate XOR to a error threshold of 0.002. It then attempts to predict what 0,0 should be.

  var test = new NeuralNetwork(2, 1, 0.3, 0.1);
  test.addLayer(2);
  test.trainSet([[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]], [[0.0],[1.0],[1.0],[1.0]], 0.002, 1000000000);
  test.predict([0.0,0.0]); // outputs a float (hopefully) close to 0.0
