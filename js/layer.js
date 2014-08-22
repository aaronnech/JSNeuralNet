/**
 * A Layer is a layer of Neurons that can be
 * constructed together into a network.
 * @param {int} neuronCount  The number of neurons in this layer.
 * @param {Layer} previousLayer The layer before this layer
 *                              null if this is the first layer.
 */
function Layer(neuronCount, previousLayer) {
	this.previousLayer_ = previousLayer;
	this.neurons_ = [];

	for (var i = 0; i < neuronCount; i++) {
		var neuronInputs = null;
		if (previousLayer) {
			neuronInputs = previousLayer.getNeurons();
		}
		var neuron = new Neuron(neuronInputs, true);
		this.neurons_.push(neuron);
	}
}


/**
 * Gets the mean squared error of all the neurons in
 * this layer.
 * @return {float} The error.
 */
Layer.prototype.getMeanSquaredError = function() {
	var errorSum = 0;
	for (var i = 0; i < this.neurons_.length; i++) {
		errorSum += Math.pow(this.neurons_[i].getError(), 2);
	}
	return errorSum / this.neurons_.length;
};


/**
 * Sets the bias of all the neurons in this layer.
 * @param {float} value The bias to set the neurons in
 * this layer to.
 */
Layer.prototype.setBias = function(value) {
	for (var i = 0; i < this.neurons_.length; i++) {
		this.neurons_[i].setBias(value);
	}
};


/**
 * Propagates error from the layer in front of this
 * layer to this layer.
 * @param  {Layer} layerInFront The layer in front of this
 *                               layer.
 */
Layer.prototype.propagateError = function(layerInFront) {
	var nextNeurons = layerInFront.getNeurons();
	for (var i = 0; i < this.neurons_.length; i++ ) {
		var myNeuron = this.neurons_[i];
		var output = myNeuron.getOutput();
		var error = 0;
		for (var j = 0; j < nextNeurons.length; j++) {
			var nextNeuronError = nextNeurons[j].getError();
			var connectionWeight = nextNeurons[j].getWeight(i);
			error += nextNeuronError * connectionWeight;
		}
		myNeuron.setError(error);
		//console.log(error * output * (1 - output));
	}
};


/**
 * Updates the weights of all the neurons in this layer.
 * @param  {float} learningRate The learning rate at which to update
 *                              the neurons in this layer.
 * @param  {float} momentum     The momentum at which to learn
 */
Layer.prototype.updateWeights = function(learningRate, momentum) {
	for (var i = 0; i < this.neurons_.length; i++) {
		this.neurons_[i].updateWeights(learningRate, momentum);
	}
};


/**
 * Synapses this layer. That is, calculates the outputs of
 * all the neurons in this layer.
 */
Layer.prototype.synapse = function() {
	for (var i = 0; i < this.neurons_.length; i++) {
		this.neurons_[i].synapse();
	}
};


/**
 * Gets all the neurons that belong to this layer.
 * @return {Array.<Neuron>} The array of neurons
 */
Layer.prototype.getNeurons = function() {
	return this.neurons_;
};


/**
 * Sets the previous layer to this layer.
 * @param {Layer} previous The previous layer.
 */
Layer.prototype.setPreviousLayer = function(previous) {
	for (var i = 0; i < this.neurons_.length; i++) {
		this.neurons_[i].setInputs(previous.getNeurons());
	}
};