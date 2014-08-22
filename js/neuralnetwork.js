/**
 * A NeuralNetwork is a collection of interconnected
 * artificial neurons that communicate via inputs and outputs.
 * A NeuralNetwork can be trained on data in order to tune synapse weights
 * to predict based on that data.
 * @param {int} inputCount   The number of inputs this network has.
 * @param {int} outputCount  The number of outputs this network has.
 * @param {float} learningRate Rate at which the network learns. A smaller rate
 *                             requires more iterations, but learns more accurately,
 *                             while a larger rate learns more quickly but less accurately.
 * @param {float} momentum   Rate at which a change propagating through
 *                           a neural network is decayed in effect
 */
function NeuralNetwork(inputCount, outputCount, learningRate, momentum) {
	this.inputCount_ = inputCount;
	this.outputCount_ = outputCount;
	this.learningRate_ = learningRate;
	this.momentum_ = momentum;

	this.inputLayer_ = new Layer(this.inputCount_, null);
	this.inputLayer_.setBias(0.0);

	this.outputLayer_ = new Layer(this.outputCount_, this.inputLayer_);

	this.hiddenLayers_ = [];
}


/**
 * Adds a layer of neurons to the network.
 * @param {int} neuronCount The number of neurons this layer contains.
 */
NeuralNetwork.prototype.addLayer = function(neuronCount) {
	var previousLayer = this.inputLayer_;
	if (this.hiddenLayers_.length > 0) {
		previousLayer = this.hiddenLayers_[this.hiddenLayers_.length - 1];
	}
	var newLayer = new Layer(neuronCount, previousLayer);
	this.hiddenLayers_.push(newLayer);
	this.outputLayer_.setPreviousLayer(newLayer);
};


/**
 * Feeds a network forward from input to output.
 * @private
 */
NeuralNetwork.prototype.forwardFeed_ = function() {
	this.inputLayer_.synapse();
	for (var i = 0; i < this.hiddenLayers_.length; i++) {
		this.hiddenLayers_[i].synapse();
	}
	this.outputLayer_.synapse();
};


/**
 * Calculates the error in the network and stores each local neuron error
 * within each neuron via backpropagation.
 * @private
 */
NeuralNetwork.prototype.calculateError_ = function(actual, expected) {
	// Output layer is unique in that,
	// it has no error to propagate back, so we
	// set it at the outer level (here)
	var outerNeurons = this.outputLayer_.getNeurons();
	for (var i = 0; i < this.outputCount_; i++) {
		var error = expected[i] - actual[i];
		// Sigmoid error propagation (derivative)
		outerNeurons[i].setError(error);
		//outerNeurons[i].setErrorDelta(error * actual[i] * (1 - actual[i]));
	}

	for (var i = this.hiddenLayers_.length - 1; i >= 0; i--) {
		var layerInFront = null;
		if (i == this.hiddenLayers_.length - 1) {
			layerInFront = this.outputLayer_;
		} else {
			layerInFront = this.hiddenLayers_[i + 1];
		}
		this.hiddenLayers_[i].propagateError(layerInFront);
	}
};


/**
 * Updates the input weights in the neural network
 * such that they minimize the propagated error stored in each neuron.
 * @private
 */
NeuralNetwork.prototype.updateWeights_ = function() {
	this.outputLayer_.updateWeights(this.learningRate_, this.momentum_);
	for (var i = 0; i < this.hiddenLayers_.length; i++) {
		this.hiddenLayers_[i].updateWeights(this.learningRate_, this.momentum_);
	}
};


/**
 * Performs the backward propagation algorithm on the network
 * to propagate error back to the input layer, and update the
 * synapse weights.
 * @param  {Array.<float>} actual  The actual values the network produced
 * @param  {Array.<float>} expected The expected values the network needs to produce
 * @private
 */
NeuralNetwork.prototype.backwardPropagate_ = function(actual, expected) {
	this.calculateError_(actual, expected);
	this.updateWeights_();
};


/**
 * Trains the network on a set of data
 * @param  {Array.<Array.<float>>} inputs   The array of training example inputs
 * @param  {Array.<Array.<float>>} expectedOutputs The array of training example outputs
 * @param  {[type]} threshold      The error (meanSquared) allowed in the network
 * @param  {[type]} maxIterations   The maximum allowed iterations the training can perform
 */
NeuralNetwork.prototype.trainSet = function(inputs, expectedOutputs, threshold, maxIterations) {
	if (inputs.length != expectedOutputs.length) {
		throw 'Invalid training set';
	}

	// Train until the network is good enough
	// (or max iterations has been reached)
	var error = 1;
	var iteration = 0;
	while (iteration < maxIterations && error > threshold) {
		var errorSum = 0;
		for (var i = 0; i < inputs.length; i++) {
			errorSum += this.train(inputs[i], expectedOutputs[i]);
		}
		var error = errorSum / inputs.length;
		iteration ++;
	}
};


/**
 * Trains on a specific input.
 * @param  {Array.<float>} input  The input data
 * @param  {Array.<float>} expectedOutput The expected output data
 * @return {float}  The MeanSquared error of this training example
 */
NeuralNetwork.prototype.train = function(input, expectedOutput) {
	if (input.length != this.inputCount_ ||
		expectedOutput.length != this.outputCount_) {
		throw 'Invalid training data or expected output dimensions.';
	}

	// Set Input values
	var inputs = this.inputLayer_.getNeurons();
	for (var i = 0; i < this.inputCount_; i++) {
		var neuron = inputs[i];
		neuron.setBias(input[i]);
	}

	// Forward feed the network
	this.forwardFeed_();


	// Backwards propagate with known outputs
	// This mutates the neural input weights
	this.backwardPropagate_(this.getOutputs(), expectedOutput);

	return this.outputLayer_.getMeanSquaredError();
};


/**
 * Predicts a output for a given input.
 * @param  {Array.<float>} input The input data to the network
 * @return {Array.<float>} The output data from the network
 */
NeuralNetwork.prototype.predict = function(input) {
	if (input.length != this.inputCount_) {
		throw 'Invalid data dimensions.';
	}

	// Set Input values
	var inputs = this.inputLayer_.getNeurons();
	for (var i = 0; i < this.inputCount_; i++) {
		var neuron = inputs[i];
		neuron.setBias(input[i]);
	}

	// Forward feed the network
	this.forwardFeed_();

	return this.getOutputs();
};


/**
 * Gets the current data that the output neurons at outputting.
 * @return {Array.<float>} The output data from the network
 */
NeuralNetwork.prototype.getOutputs = function() {
	return this.outputLayer_.getNeurons().map(function(neuron) {
		return neuron.getOutput();
	});
};